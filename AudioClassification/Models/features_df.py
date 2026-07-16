""" 
This file implement methods to build feature table for random forest, Xgboost models

Do batch training to handle dimensionality
"""
"""
Full feature extraction pipeline for audio event classification
(gunshot / explosion / drone / other).

Covers:
  - Spectral shape: centroid, bandwidth, rolloff, flatness, flux
  - Temporal/envelope: ZCR, RMS (+variance), attack time, decay time,
    temporal centroid, crest factor
  - Harmonic/pitch: F0 (mean/std), voiced fraction, harmonic-to-noise ratio
  - Band energy ratios: low/mid/high + a rotor-fundamental band tuned for
    drone blade-pass frequencies

Requires: librosa, numpy, pandas
    pip install librosa numpy pandas
"""

import logging

import librosa
import numpy as np
import pandas as pd
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------- shared helper -----------------
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

def summarize(feat_matrix, name):
    """Collapse a 1D (n_frames,) frame array into named
    mean/std/min/max summary stats."""
    feat_matrix = np.atleast_1d(feat_matrix)
    return {
        f"{name}_mean": float(np.mean(feat_matrix)),
        f"{name}_std": float(np.std(feat_matrix)),
        f"{name}_min": float(np.min(feat_matrix)),
        f"{name}_max": float(np.max(feat_matrix)),
    }


def summarize_matrix(feat_matrix, name):
    """Collapse a 2D (n_coeffs, n_frames) matrix into named per-coefficient
    mean/std/min/max stats, e.g. mfcc_0_mean, mfcc_0_std, ..., mfcc_12_max.
    Used for MFCCs / delta / delta-delta, matching your original per-file
    coefficient naming convention."""
    means = np.mean(feat_matrix, axis=1)
    stds = np.std(feat_matrix, axis=1)
    mins = np.min(feat_matrix, axis=1)
    maxs = np.max(feat_matrix, axis=1)

    feats = {}
    for i in range(feat_matrix.shape[0]):
        feats[f"{name}_{i}_mean"] = float(means[i])
        feats[f"{name}_{i}_std"] = float(stds[i])
        feats[f"{name}_{i}_min"] = float(mins[i])
        feats[f"{name}_{i}_max"] = float(maxs[i])
    return feats


# ----------------- spectral shape descriptors -----------------
def extract_spectral_features(S, sr, roll_percent=0.85):
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=roll_percent)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    # Spectral flux: L2 norm of frame-to-frame diff of normalized magnitude spectrum
    S_norm = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-10)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0.0], flux])

    feats = {}
    feats.update(summarize(centroid, "centroid"))
    feats.update(summarize(bandwidth, "bandwidth"))
    feats.update(summarize(rolloff, "rolloff"))
    feats.update(summarize(flatness, "flatness"))
    feats.update(summarize(flux, "flux"))
    feats["flux_peak_to_mean_ratio"] = float(np.max(flux) / (np.mean(flux) + 1e-10))
    return feats


# ----------------- temporal / envelope descriptors -----------------
def extract_temporal_features(y, sr, frame_length=2048, hop_length=512, threshold_db=-20.0):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    feats = {}
    feats.update(summarize(zcr, "zcr"))
    feats.update(summarize(rms, "rms"))

    # Attack / decay / temporal centroid, derived from the RMS envelope
    peak_idx = int(np.argmax(rms))
    peak_time = times[peak_idx]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) + 1e-10)

    above_before_peak = np.where(rms_db[: peak_idx + 1] >= threshold_db)[0]
    onset_idx = above_before_peak[0] if len(above_before_peak) > 0 else 0
    attack_time = peak_time - times[onset_idx]

    below_after_peak = np.where(rms_db[peak_idx:] <= threshold_db)[0]
    decay_idx = peak_idx + below_after_peak[0] if len(below_after_peak) > 0 else len(rms) - 1
    decay_time = times[decay_idx] - peak_time

    energy = rms ** 2
    temporal_centroid = float(np.sum(times * energy) / (np.sum(energy) + 1e-10))

    crest_factor = float(np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-10))

    feats["attack_time"] = float(attack_time)
    feats["decay_time"] = float(decay_time)
    feats["temporal_centroid"] = temporal_centroid
    feats["crest_factor"] = crest_factor
    return feats


# ----------------- harmonic / pitch descriptors -----------------
def extract_harmonic_features(y, sr, fmin=50.0, fmax=2000.0):
    feats = {}
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
        valid = f0[~np.isnan(f0)]
        feats["f0_mean"] = float(np.mean(valid)) if len(valid) > 0 else 0.0
        feats["f0_std"] = float(np.std(valid)) if len(valid) > 0 else 0.0
        feats["voiced_fraction"] = float(np.mean(voiced_flag))
    except Exception as e:
        logger.warning(f"pyin F0 extraction failed ({e}); filling zeros")
        feats["f0_mean"] = 0.0
        feats["f0_std"] = 0.0
        feats["voiced_fraction"] = 0.0

    # Harmonic-to-noise ratio proxy via harmonic/percussive source separation:
    # a strong steady tone (drone) -> mostly harmonic energy -> high HNR.
    # Noisy/impulsive sounds (gunshot/explosion) -> mostly percussive/noise energy -> low HNR.
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(y_harm ** 2)
    perc_energy = np.sum(y_perc ** 2)
    feats["hnr_db"] = float(10 * np.log10((harm_energy + 1e-10) / (perc_energy + 1e-10)))
    return feats


# ----------------- MFCC / delta / delta-delta descriptors -----------------
def extract_mfcc_features(y, sr, n_mfcc=13, n_mels=128):
    """Log-mel-based MFCCs plus their first and second derivatives
    (velocity / acceleration of the timbral envelope)."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    mfccs = librosa.feature.mfcc(S=log_mel, n_mfcc=n_mfcc)
    delta_mfccs = librosa.feature.delta(mfccs, order=1)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    feats = {}
    feats.update(summarize_matrix(mfccs, "mfcc"))
    feats.update(summarize_matrix(delta_mfccs, "delta"))
    feats.update(summarize_matrix(delta2_mfccs, "delta2"))
    return feats


# ----------------- band energy ratios -----------------
def extract_band_energy_features(S, sr, n_fft):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = S ** 2
    total_energy = np.sum(power) + 1e-10

    # low/mid/high are generic broadband splits; rotor_fundamental targets the
    # 80-300 Hz range where multirotor drone blade-pass fundamentals typically sit
    band_defs = {
        "band_low": (0, 300),
        "band_mid": (300, 2000),
        "band_high": (2000, sr / 2),
        "band_rotor_fundamental": (80, 300),
    }

    feats = {}
    for name, (lo, hi) in band_defs.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_energy = np.sum(power[mask, :])
        feats[f"{name}_ratio"] = float(band_energy / total_energy)
    return feats


# ----------------- per-file feature row -----------------
def extract_all_features(path, n_fft=2048, hop_length=512, sr=None, n_mfcc=13, n_mels=128):
    """Load one audio file and compute the full feature dict for it."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    feats = {}
    feats.update(extract_mfcc_features(y, sr, n_mfcc=n_mfcc, n_mels=n_mels))
    feats.update(extract_spectral_features(S, sr))
    feats.update(extract_temporal_features(y, sr, frame_length=n_fft, hop_length=hop_length))
    feats.update(extract_harmonic_features(y, sr))
    feats.update(extract_band_energy_features(S, sr, n_fft))
    return feats


# ----------------- DataFrame builder -----------------
def build_feature_table(data_dir, split, n_fft=2048, hop_length=512, sr=None,
                         n_mfcc=13, n_mels=128):
    """
    Build a pandas DataFrame of summary features, one row per audio file.

    Parameters
    ----------
    split: str "train" or "test"
    audio_files : list of str
        Paths to audio files.
    labels : list, optional
        Class label per file (same order/length as audio_files). If given,
        a 'label' column is added.
    n_fft, hop_length : int
        STFT parameters shared across all spectral/band features.
    sr : int, optional
        Force a target sample rate; None keeps each file's native rate.
    n_mfcc, n_mels : int
        MFCC coefficient count and mel filterbank size.

    Returns
    -------
    pd.DataFrame
    """
    audio_files, labels, class_2_indx = load_toc(data_dir, split)
    logger.info(class_2_indx) #Label encoding map

    if labels is not None and len(labels) != len(audio_files):
        raise ValueError("labels must be the same length as audio_files")

    rows = []
    for i, path in enumerate(audio_files):
        logger.info(f"Extracting features [{i + 1}/{len(audio_files)}]: {path}")
        try:
            row = extract_all_features(path, n_fft=n_fft, hop_length=hop_length, sr=sr,
                                        n_mfcc=n_mfcc, n_mels=n_mels)
            row["filepath"] = path
            if labels is not None:
                row["label"] = labels[i]
            rows.append(row)
        except Exception as e:
            logger.error(f"Failed to extract features for {path}: {e}")

    df = pd.DataFrame(rows)

    # keep filepath/label as leading columns if present
    leading = [c for c in ("filepath", "label") if c in df.columns]
    other = [c for c in df.columns if c not in leading]
    return df[leading + other]

        

if __name__ == "__main__":
    pass

""" 
Feature Cheat Sheet

MFCC Features (Timbre)
| Feature         | Measures                                      | High Value             | Low Value          | Useful For                   |
| --------------- | --------------------------------------------- | ---------------------- | ------------------ | ---------------------------- |
| **MFCC**        | Overall spectral envelope (sound "character") | Complex timbre         | Simple timbre      | General audio classification |
| **Delta MFCC**  | Rate of change of MFCC                        | Rapidly changing sound | Stable sound       | Gunshots, explosions         |
| **Delta² MFCC** | Acceleration of MFCC change                   | Abrupt transitions     | Smooth transitions | Transient events             |


Spectral Features (Frequency Count)
| Feature                  | Measures                                    | High Value Means                    | Low Value Means        |
| ------------------------ | ------------------------------------------- | ----------------------------------- | ---------------------- |
| **Spectral Centroid**    | Center of spectral energy                   | Bright sound, more high frequencies | Dark/bassy sound       |
| **Spectral Bandwidth**   | Spread of frequencies                       | Wide frequency range                | Narrow frequency range |
| **Spectral Rolloff**     | Frequency containing 85% of energy          | More high-frequency energy          | Mostly low frequencies |
| **Spectral Flatness**    | Noise-like vs tonal                         | Noise                               | Pure tone              |
| **Spectral Flux**        | Change between consecutive spectra          | Rapid frequency changes             | Stable spectrum        |
| **Flux Peak/Mean Ratio** | Largest spectral change relative to average | One large transient                 | Constant behavior      |


Temporal Features (Time Domain)
| Feature                      | Measures                           | High Value           | Low Value             |
| ---------------------------- | ---------------------------------- | -------------------- | --------------------- |
| **Zero Crossing Rate (ZCR)** | Number of sign changes             | Noisy/high-frequency | Smooth/low-frequency  |
| **RMS Energy**               | Loudness                           | Loud                 | Quiet                 |
| **Attack Time**              | Time to reach peak amplitude       | Slow onset           | Instant onset         |
| **Decay Time**               | Time to fade after peak            | Long ringing         | Short impulse         |
| **Temporal Centroid**        | Center of energy in time           | Energy occurs later  | Energy occurs earlier |
| **Crest Factor**             | Peak amplitude relative to average | Sharp impulse        | Steady sound          |


Harmonic Features (Pitch)
| Feature                           | Measures                        | High Value     | Low Value    |
| --------------------------------- | ------------------------------- | -------------- | ------------ |
| **F0 Mean**                       | Average fundamental frequency   | High pitch     | Low pitch    |
| **F0 Std**                        | Pitch variation                 | Variable pitch | Stable pitch |
| **Voiced Fraction**               | Fraction with detectable pitch  | Mostly tonal   | Mostly noise |
| **HNR (Harmonic-to-Noise Ratio)** | Harmonic energy vs noisy energy | Tonal          | Noisy        |


Band Energy Ratios
| Feature        | Frequency Range | Indicates                    |
| -------------- | --------------- | ---------------------------- |
| **Low Band**   | 0–300 Hz        | Engines, drones, bass        |
| **Rotor Band** | 80–300 Hz       | Drone blade-pass frequencies |
| **Mid Band**   | 300–2000 Hz     | Speech, machinery            |
| **High Band**  | >2000 Hz        | Gunshots, glass, explosions  |


NOTE:
Since most features are computed for every frame, the code summarizes them.

Statistic	Meaning
Mean	Average value
Std	Variation over time
Min	Smallest observed value
Max	Largest observed value
"""