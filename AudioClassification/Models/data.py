#Notes
""" 
Notes:
- Will be working with the data in the frequency domain. 
Features:
- MFCCs
- spectral features 
- Mel spectogram
- Bark scale features
- Loudness (A-Weighted) , percieved loudedness
- Sharpness, roughness, fluctuation strength


Practical Tips

Always use log-mel as your baseline — it beats raw MFCCs on almost every ESC benchmark

Data augmentation is critical: time stretching, pitch shifting, mixup, SpecAugment
Pretrained models (PANNs trained on AudioSet) give huge boosts via transfer learning

For real-time / edge applications, stick to handcrafted features (MFCCs + spectral) — they're fast and interpretable
Consider ensemble of mel spectrogram CNN + handcrafted feature classifier for robustness
"""
""" 

Steps
1. Get a json with the following structure with all raw audios

{
audio_file: [], "list of filepaths"
audio_label: [], "animals, gunshots, explosion, human_voices, uav, unknown"
}

2. Evaluation will be done using batch cross validation

3. Construct model input matrix
- Each row will be uniform size, 10 sec clip
- Features:
    - Log-Mel
    - Delta Mel

Note:
Delta Mel is computed directly from log mel spectogram
So Why Do People Still Use Them?
Because CNNs don't automatically learn temporal derivatives from a static 2D image — stacking deltas is an inductive bias shortcut:

A conv kernel could learn to compute differences across the time axis on its own
But explicitly providing deltas makes that gradient immediately available, speeding up learning and improving sample efficiency
Particularly helpful when your dataset is small — the CNN doesn't have to discover temporal dynamics from scratch

"""

import random
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from collections import deque
import logging
import random
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AudioBootstrapper:
    def __init__(self):
        pass

    def BootStrap(self, og_folder_path: Path, sub_folder_names: list[str], output_folder_path: Path,
                  overlay_folder_name: str, aug_technique_num: int, num_bootstraps: int):
        """
        og_folder_path: path to the original audio files (Train or Test parent directories in Raw_Audios folder for Data_v2)
        output_folder_path: path to the save location (Bootstrap Folder Train or Test)
        aug_technique_num: number of audio augmentation techniques applied
        sub_folder_names: Explosion, Gunshots, shortened_unlabeled, etc

        num_bootstraps: number of bootstrap files created
        """
        og_folder_path = Path(og_folder_path)
        output_folder_path = Path(output_folder_path)
        output_folder_path.mkdir(parents=True, exist_ok=True)

        # Table of content to keep track of audio labels
        toc = {
            "file_name": [],
            "label": [],  # "Explosion", "UAV", "Gunshots", "unlabeled", etc
        }

        # File paths that can be used to overlay with original audio
        all_files_overlay = self._list_files_in_subdirs(og_folder_path, [overlay_folder_name])

        if aug_technique_num > 5:
            aug_technique_num = 5
            logger.warning("Max techniques is 5. 'aug_technique_num' was set to 5")

        if aug_technique_num <= 0:
            aug_technique_num = 1
            logger.warning("Min technique is 1. 'aug_technique_num' was set to 1")

        file_cntr = 0
        for sub_folder in sub_folder_names:
            # Get a list of all the original source file paths
            all_files_og = self._list_files_in_subdirs(og_folder_path, [sub_folder])

            og_file_deq = deque(all_files_og)
            for i in range(num_bootstraps):
                if not og_file_deq:
                    og_file_deq = deque(all_files_og)
                file_2_aug = og_file_deq[-1]
                y, sr = librosa.load(file_2_aug)

                # Applying audio augmentations
                for a in range(aug_technique_num):
                    aug_tech_encoder = random.randint(1, 5)
                    if aug_tech_encoder == 1:
                        y = self._aug_time_stretch(y, sr)
                    elif aug_tech_encoder == 2:
                        y = self._aug_pitch_shift(y, sr)
                    elif aug_tech_encoder == 3:
                        y = self._aug_gaussian_noise(y)
                    elif aug_tech_encoder == 4:
                        y = self._aug_time_shift(y, sr)
                    elif aug_tech_encoder == 5:
                        y = self._aug_overlay(y, sr, all_files_overlay)

                og_file_deq.pop()  # eliminating file from choices

                # save new audio
                file_cntr += 1
                logger.info(f"Files processed: {file_cntr}/{num_bootstraps*len(sub_folder_names)}")
                full_out_path = output_folder_path / f"audio_{file_cntr}.wav"
                sf.write(full_out_path, y, sr)

                toc["file_name"].append(f"audio_{file_cntr}.wav")
                toc["label"].append(sub_folder)

        # Save table of contents as JSON in the parent of the output folder
        toc_path = output_folder_path.parent / "toc.json"
        with open(toc_path, "w") as f:
            json.dump(toc, f, indent=2)

        return toc

    # --------------- Helper Functions ------------------------
    def _list_files_in_subdirs(self, parent_dir: str | Path, subdir_names: list[str]) -> list[Path]:
        """
        Given a parent directory and a list of subdirectory names,
        return a list of Path objects for all files inside each subdirectory
        (non-recursive — only direct contents of each subdir).
        """
        parent_dir = Path(parent_dir)
        all_files = []

        for name in subdir_names:
            subdir = parent_dir / name
            if not subdir.is_dir():
                logger.warning(f"{subdir} is not a valid directory, skipping.")
                continue
            all_files.extend(p for p in subdir.iterdir() if p.is_file())

        return all_files

    # ------------------------- Audio Augmentation Techniques -----------------------------
    # Total 5 augmentation techniques

    # Changes speed of audio, changes rate by a random factor between .85 and 1.15
    def _aug_time_stretch(self, y, sr, rate=None):
        rate = rate or random.uniform(0.85, 1.15)
        try:
            return librosa.effects.time_stretch(y, rate=rate)
        except Exception:
            logger.warning("timestretch_failed")
            return y

    # Changes pitch of the audio
    def _aug_pitch_shift(self, y, sr, n_steps=None):
        n_steps = n_steps if n_steps is not None else random.uniform(-2.5, 2.5)
        try:
            return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        except Exception:
            logger.warning("pitchshift_failed")
            return y

    # Adds random background noise to audio
    def _aug_gaussian_noise(self, y, snr_db=None):
        snr_db = snr_db if snr_db is not None else random.uniform(10, 25)
        sig_power = np.mean(y ** 2) + 1e-12
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
        return y + noise

    # Moves where the sound starts within the clip without changing pitch or speed
    def _aug_time_shift(self, y, sr, max_shift_sec=0.3):
        max_shift = int(sr * max_shift_sec)
        if max_shift <= 0:
            return y
        shift = random.randint(-max_shift, max_shift)
        return np.roll(y, shift)

    # Overlays another audio
    def _aug_overlay(self, y, sr, overlay_files, snr_db=None):
        """Mix in a random clip from the overlay folder at a random SNR."""
        if not overlay_files:
            logger.warning("overlay_skipped")
            return y
        snr_db = snr_db if snr_db is not None else random.uniform(3, 15)
        overlay_path = random.choice(overlay_files)
        try:
            noise, _ = librosa.load(overlay_path, sr=sr)
        except Exception:
            logger.warning("overlay_failed")
            return y

        if len(noise) == 0:
            logger.warning("overlay_failed")
            return y

        # Loop or trim the overlay clip to match target length
        if len(noise) < len(y):
            reps = int(np.ceil(len(y) / len(noise)))
            noise = np.tile(noise, reps)
        start = random.randint(0, max(0, len(noise) - len(y)))
        noise = noise[start:start + len(y)]
        if len(noise) < len(y):
            noise = np.pad(noise, (0, len(y) - len(noise)))

        sig_power = np.mean(y ** 2) + 1e-12
        noise_power = np.mean(noise ** 2) + 1e-12
        desired_noise_power = sig_power / (10 ** (snr_db / 10))
        noise = noise * np.sqrt(desired_noise_power / noise_power)

        return y + noise


if __name__ == "__main__":
    pass
