"""
audio_cnn_classifier.py
========================
Audio classification with a CNN over mel-spectrograms.

Designed to be imported as a module (e.g. from a Jupyter notebook) rather
than run as a script — no module-level global state. All settings live in
an AudioConfig instance that you create and pass around explicitly.

Assumes audio files are already uniform, fixed length (pre-chunked upstream).
Each file is used in full and mapped to exactly one training example.

Install dependencies:
    pip install torch torchaudio librosa numpy scikit-learn

Expected data layout (matches train/test + *_toc.json metadata):
    data/
        train/
            file1.wav
            file2.wav
            ...
        test/
            file1.wav
            ...
        train_toc.json   # {"file_name": [...], "label": [...]}
        test_toc.json     # {"file_name": [...], "label": [...]}

Typical notebook usage:
    from audio_cnn_classifier import AudioConfig, train

    cfg = AudioConfig(data_dir="data", duration=3.0, epochs=15)
    model, class_to_idx, history = train(cfg)
"""

import os
import json
from dataclasses import dataclass

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ----------------------------
# Config
# ----------------------------
@dataclass
class AudioConfig:
    data_dir: str = "data"        # root folder containing train/, test/, train_toc.json, test_toc.json
    sample_rate: int = 16000       # resample everything to this rate
    duration: float = None         # seconds; fixed clip length used for padding/truncation.
                                    # Set to None to auto-detect: train() will scan your audio
                                    # files and use the longest clip's length as the uniform size.
    n_mels: int = 128              # mel bands (frequency resolution)
    n_fft: int = 1024
    hop_length: int = 256
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    val_split: float = 0.15        # fraction of train set held out for validation
    num_workers: int = 2
    checkpoint_path: str = "best_audio_cnn.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# 1. Build file list + labels from *_toc.json
# ----------------------------
def load_toc(data_dir, split, class_to_idx=None):
    """
    split: "train" or "test"
    Reads data/{split}_toc.json with structure {"file_name": [...], "label": [...]}
    and resolves each file_name to data/{split}/{file_name}.

    class_to_idx is built from the train split and passed in for test,
    so label indices stay consistent across both.
    """
    toc_path = os.path.join(data_dir, f"{split}_toc.json")
    with open(toc_path, "r") as f:
        toc = json.load(f)

    file_names = toc["file_name"]
    raw_labels = toc["label"]

    if class_to_idx is None:
        classes = sorted(set(raw_labels))
        class_to_idx = {c: i for i, c in enumerate(classes)}

    files = [os.path.join(data_dir, split, fn) for fn in file_names]
    labels = [class_to_idx[l] for l in raw_labels]

    return files, labels, class_to_idx


def compute_max_duration(files):
    """
    Scans a list of audio file paths and returns the duration (in seconds)
    of the longest one. Uses librosa.get_duration(path=...), which reads
    file metadata/headers rather than decoding the full waveform, so this
    is fast even over thousands of files.

    Useful when your files are NOT uniform length and you want to pad
    everything up to the length of the longest clip instead of picking
    an arbitrary fixed duration.
    """
    max_duration = 0.0
    for path in files:
        try:
            d = librosa.get_duration(path=path)
        except Exception as e:
            print(f"Warning: could not read duration for {path} ({e}); skipping")
            continue
        if d > max_duration:
            max_duration = d
    return max_duration


# ----------------------------
# 2. Feature extraction: audio -> mel-spectrogram
# ----------------------------
def audio_to_melspec(path, config: AudioConfig):
    y, _ = librosa.load(path, sr=config.sample_rate)

    # Since files are assumed uniform length, this pad/truncate is just a
    # safety net for tiny mismatches (e.g. off-by-one-sample rounding),
    # not a routine truncation of real content.
    target_len = int(config.sample_rate * config.duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize to roughly [-1, 1]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    return mel_db.astype(np.float32)  # shape: (n_mels, time_steps)


# ----------------------------
# 3. Dataset
# ----------------------------
class AudioDataset(Dataset):
    def __init__(self, files, labels, config: AudioConfig, augment=False):
        super().__init__()
        self.files = files
        self.labels = labels
        self.config = config
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = audio_to_melspec(self.files[idx], self.config)

        if self.augment:
            mel = self._spec_augment(mel)

        mel = torch.tensor(mel).unsqueeze(0)  # add channel dim -> (1, n_mels, time)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel, label

    @staticmethod
    def _spec_augment(mel, freq_mask=12, time_mask=20):
        mel = mel.copy()
        # clamp mask sizes to the actual spectrogram shape so this never
        # errors out if n_mels/time steps end up smaller than the mask sizes
        freq_mask = min(freq_mask, max(1, mel.shape[0] - 1))
        time_mask = min(time_mask, max(1, mel.shape[1] - 1))

        f0 = np.random.randint(0, mel.shape[0] - freq_mask + 1)
        mel[f0:f0 + freq_mask, :] = 0
        t0 = np.random.randint(0, mel.shape[1] - time_mask + 1)
        mel[:, t0:t0 + time_mask] = 0
        return mel


# ----------------------------
# 4. CNN model
# ----------------------------
class AudioCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),  # collapse to fixed size regardless of input length
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


# ----------------------------
# 5. Train / eval loops
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(mel)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += mel.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        out = model(mel)
        loss = criterion(out, labels)
        total_loss += loss.item() * mel.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += mel.size(0)
    return total_loss / total, correct / total


# ----------------------------
# 6. Top-level training routine
# ----------------------------
def train(config: AudioConfig):
    """
    Runs the full pipeline: load toc files, split, train, evaluate on test.
    Returns (model, class_to_idx, history) so a notebook can inspect/plot
    results without re-running anything.
    """
    device = torch.device(config.device)

    all_train_files, all_train_labels, class_to_idx = load_toc(config.data_dir, "train")
    print(f"Train split: {len(all_train_files)} files across {len(class_to_idx)} classes: {class_to_idx}")

    test_files, test_labels, _ = load_toc(config.data_dir, "test", class_to_idx=class_to_idx)
    print(f"Test split: {len(test_files)} files")

    if config.duration is None:
        print("config.duration is None -> scanning audio files to find max length...")
        config.duration = compute_max_duration(all_train_files + test_files)
        print(f"Using duration={config.duration:.3f}s (longest clip found) as the uniform padding length")

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_train_files, all_train_labels,
        test_size=config.val_split, stratify=all_train_labels, random_state=42
    )

    train_ds = AudioDataset(train_files, train_labels, config, augment=True)
    val_ds = AudioDataset(val_files, val_labels, config, augment=False)
    test_ds = AudioDataset(test_files, test_labels, config, augment=False)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = AudioCNN(n_classes=len(class_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{config.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.checkpoint_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(config.checkpoint_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    return model, class_to_idx, history


if __name__ == "__main__":
    cfg = AudioConfig()
    train(cfg)