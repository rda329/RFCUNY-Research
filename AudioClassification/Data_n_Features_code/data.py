"""
This file handles bootstrapping of audio data.
"""

import random
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from collections import deque
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AudioBootstrapper:
    def __init__(self):
        pass

    def BootStrap_v2(self, og_folder_path: Path, sub_folder_names: list[str], output_folder_path: Path,
                      aug_technique_num: int, num_bootstraps: int, toc_name: str, allow_self_overlay: bool = True):
        """
        Same as BootStrap, but overlay audio can be drawn from ANY class in
        `sub_folder_names` instead of a single fixed overlay folder.

        og_folder_path: path to the original audio files (Train or Test parent dir)
        output_folder_path: path to the save location (Bootstrap Folder Train or Test)
        sub_folder_names: classes to bootstrap AND the pool of classes eligible for overlay
                        (e.g. Explosion, Gunshots, UAV, shortened_unlabeled, etc)
        aug_technique_num: number of audio augmentation techniques applied per file
        num_bootstraps: number of bootstrap files created per class
        allow_self_overlay: if False, a class will never be overlaid with audio from
                            its own class (falls back to allowing it if no other
                            classes have files available)

        toc additionally contains one binary column per class, "overlay_<class_name>",
        indicating whether that class's audio was used as an overlay source for the
        corresponding output file.
        """
        og_folder_path = Path(og_folder_path)
        output_folder_path = Path(output_folder_path)
        output_folder_path.mkdir(parents=True, exist_ok=True)

        if aug_technique_num > 5:
            aug_technique_num = 5
            logger.warning("Max techniques is 5. 'aug_technique_num' was set to 5")

        if aug_technique_num <= 0:
            aug_technique_num = 1
            logger.warning("Min technique is 1. 'aug_technique_num' was set to 1")

        # Pre-list files for every class once, reused both as bootstrap source and overlay pool
        class_files = {
            name: self._list_files_in_subdirs(og_folder_path, [name])
            for name in sub_folder_names
        }

        # Table of content to keep track of audio labels + which classes were overlaid in
        toc = {
            "file_name": [],
            "base_audio": [],  # "Explosion", "UAV", "Gunshots", "unlabeled", etc
        }
        for name in sub_folder_names:
            toc[f"overlay_{name}"] = []

        file_cntr = 0
        total_files = num_bootstraps * len(sub_folder_names)

        for sub_folder in sub_folder_names:
            all_files_og = class_files[sub_folder]
            if not all_files_og:
                logger.warning(f"No files found for class '{sub_folder}', skipping.")
                continue

            og_file_deq = deque(all_files_og)
            for i in range(num_bootstraps):
                if not og_file_deq:
                    og_file_deq = deque(all_files_og)
                file_2_aug = og_file_deq[-1]
                y, sr = librosa.load(file_2_aug)

                # Track which classes were used as overlay sources for this file
                overlay_used = {name: 0 for name in sub_folder_names}

                # Applying audio augmentations (5 techniques available, incl. overlay)
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
                        overlay_classes = self._pick_overlay_classes(
                            sub_folder_names, class_files, sub_folder, allow_self_overlay
                        )
                        if not overlay_classes:
                            logger.warning("overlay_skipped_no_candidates_or_zero_chosen")
                            continue
                        # Ensure the base clip is at least 1 minute long before overlaying
                        y = self._pad_to_min_duration(y, sr, min_duration_sec=60)
                        for overlay_class in overlay_classes:
                            y = self._aug_overlay(y, sr, class_files[overlay_class])
                            overlay_used[overlay_class] = 1

                og_file_deq.pop()  # eliminating file from choices

                # save new audio
                file_cntr += 1
                logger.info(f"Files processed: {file_cntr}/{total_files}")
                full_out_path = output_folder_path / f"audio_{file_cntr}.wav"
                sf.write(full_out_path, y, sr)

                toc["file_name"].append(f"audio_{file_cntr}.wav")
                toc["base_audio"].append(sub_folder)
                for name in sub_folder_names:
                    toc[f"overlay_{name}"].append(overlay_used[name])

        # Save table of contents as JSON in the parent of the output folder
        toc_path = output_folder_path.parent / f"{toc_name}.json"
        with open(toc_path, "w") as f:
            json.dump(toc, f, indent=2)

        return toc

    def BootStrap_v1(self, og_folder_path: Path, sub_folder_names: list[str], output_folder_path: Path,
                  aug_technique_num: int, num_bootstraps: int, toc_name: str):
        """
        Bootstraps audio files using non-overlay augmentation techniques only
        (time stretch, pitch shift, gaussian noise, time shift). For overlay-based
        bootstrapping, use BootStrap_v2 instead.

        og_folder_path: path to the original audio files (Train or Test parent directories in Raw_Audios folder for Data_v2)
        output_folder_path: path to the save location (Bootstrap Folder Train or Test)
        sub_folder_names: Explosion, Gunshots, shortened_unlabeled, etc
        aug_technique_num: number of audio augmentation techniques applied
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

        if aug_technique_num > 4:
            aug_technique_num = 4
            logger.warning("Max techniques is 4. 'aug_technique_num' was set to 4")

        if aug_technique_num <= 0:
            aug_technique_num = 1
            logger.warning("Min technique is 1. 'aug_technique_num' was set to 1")

        file_cntr = 0
        for sub_folder in sub_folder_names:
            # Get a list of all the original source file paths
            all_files_og = self._list_files_in_subdirs(og_folder_path, [sub_folder])
            if not all_files_og:
                logger.warning(f"No files found for class '{sub_folder}', skipping.")
                continue

            og_file_deq = deque(all_files_og)
            for i in range(num_bootstraps):
                if not og_file_deq:
                    og_file_deq = deque(all_files_og)
                file_2_aug = og_file_deq[-1]
                y, sr = librosa.load(file_2_aug)

                # Applying audio augmentations (4 non-overlay techniques only)
                for a in range(aug_technique_num):
                    aug_tech_encoder = random.randint(1, 4)
                    if aug_tech_encoder == 1:
                        y = self._aug_time_stretch(y, sr)
                    elif aug_tech_encoder == 2:
                        y = self._aug_pitch_shift(y, sr)
                    elif aug_tech_encoder == 3:
                        y = self._aug_gaussian_noise(y)
                    elif aug_tech_encoder == 4:
                        y = self._aug_time_shift(y, sr)

                og_file_deq.pop()  # eliminating file from choices

                # save new audio
                file_cntr += 1
                logger.info(f"Files processed: {file_cntr}/{num_bootstraps * len(sub_folder_names)}")
                full_out_path = output_folder_path / f"audio_{file_cntr}.wav"
                sf.write(full_out_path, y, sr)

                toc["file_name"].append(f"audio_{file_cntr}.wav")
                toc["label"].append(sub_folder)

        # Save table of contents as JSON in the parent of the output folder
        toc_path = output_folder_path.parent / f"{toc_name}.json"
        with open(toc_path, "w") as f:
            json.dump(toc, f, indent=2)

        return toc

    # --------------- Helper Functions ------------------------
    def _pad_to_min_duration(self, y: np.ndarray, sr: int, min_duration_sec: float = 60) -> np.ndarray:
        """
        Zero-pads (silence) the end of `y` so its duration is at least
        `min_duration_sec` seconds. If `y` is already long enough, it is
        returned unchanged.
        """
        min_samples = int(sr * min_duration_sec)
        if len(y) >= min_samples:
            return y
        pad_amount = min_samples - len(y)
        return np.pad(y, (0, pad_amount), mode="constant")

    def _pick_overlay_classes(self, sub_folder_names: list[str], class_files: dict, base_class: str,
                               allow_self_overlay: bool) -> list[str]:
        """
        Pick a uniformly random NUMBER of overlay classes (0 to the max number of
        eligible classes), then sample that many distinct classes without replacement
        to draw overlay clips from. Restricted to classes that actually have files
        available. Optionally excludes the base class itself.

        Returns an empty list if 0 overlays were rolled, or if no candidates exist.
        """
        candidates = [name for name in sub_folder_names if class_files.get(name)]

        if not allow_self_overlay:
            non_self_candidates = [name for name in candidates if name != base_class]
            if non_self_candidates:
                candidates = non_self_candidates
            # else: fall back to allowing self-overlay since nothing else is available

        if not candidates:
            return []

        max_overlays = len(candidates)
        num_overlays = random.randint(0, max_overlays)  # uniform over [0, max_overlays]

        if num_overlays == 0:
            return []

        return random.sample(candidates, num_overlays)

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
    # Total 5 augmentation techniques (4 non-overlay + overlay, overlay only used in v2)

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

    # Overlays another audio (used only by BootStrap_v2)
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
