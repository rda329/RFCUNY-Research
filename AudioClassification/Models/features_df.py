""" 
This file implement methods to build feature table for random forest, Xgboost models
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
def load_toc(data_dir, split, class_to_idx=None, base_to_idx=None):
    """
    split: "train" or "test"
    Reads data/{split}_toc.json.
 
    Supports two TOC formats:
      1. Original (single-label):
         {"file_name": [...], "label": [...]}
 
      2. Base + overlay (multi-label):
         {"file_name": [...],
          "base_audio": ["Airplanes", "Drones", ...],       # primary source, categorical
          "overlay_Airplanes": [0,1,...], "overlay_Drones": [...], ...}  # 0/1 event flags
         Any column whose values are all 0/1 is treated as an overlay class.
         The one remaining non-binary, non-file_name column is treated as
         the categorical base class.
 
    class_to_idx : dict, optional
        Mapping for overlay classes (e.g. built from train, passed in for test
        so indices stay consistent across splits). Ignored for "single" format
        TOCs (use class_to_idx there too, same param, same purpose).
    base_to_idx : dict, optional
        Mapping for the base_audio class (built from train, passed in for test).
        Only used for the "base_overlay" format.
 
    Returns
    -------
    files : list of str
    labels : list
        "single"       -> list of int class indices
        "base_overlay" -> list of dicts: {"base": int or None, "overlays": {class: 0/1}}
    class_to_idx : dict
        "single"       -> label name -> index
        "base_overlay" -> overlay class name -> index (empty dict if none found)
    base_to_idx : dict
        base class name -> index (empty dict unless "base_overlay" format with a base column)
    label_format : str
        "single" or "base_overlay"
    """
    toc_path = os.path.join(data_dir, f"{split}_toc.json")
    with open(toc_path, "r") as f:
        toc = json.load(f)
 
    file_names = toc["file_name"]
    n = len(file_names)
 
    if "label" in toc:
        # ---- original single-label format ----
        label_format = "single"
        raw_labels = toc["label"]
 
        if class_to_idx is None:
            classes = sorted(set(raw_labels))
            class_to_idx = {c: i for i, c in enumerate(classes)}
 
        labels = [class_to_idx[l] for l in raw_labels]
        base_to_idx = {}
 
    else:
        # ---- base + overlay format ----
        label_format = "base_overlay"
 
        def is_binary_column(values):
            try:
                return all(int(v) in (0, 1) for v in values)
            except (ValueError, TypeError):
                return False
 
        other_cols = [k for k in toc.keys() if k != "file_name"]
        overlay_cols = [c for c in other_cols if is_binary_column(toc[c])]
        base_cols = [c for c in other_cols if c not in overlay_cols]
 
        if len(base_cols) > 1:
            logger.warning(
                f"{split}_toc.json has multiple non-binary columns "
                f"{base_cols}; only the first ({base_cols[0]}) will be used "
                f"as the base class, rest are ignored."
            )
        base_col = base_cols[0] if base_cols else None
 
        # ---- overlay classes ----
        if class_to_idx is None:
            overlay_names = [c.replace("overlay_", "", 1) for c in overlay_cols]
            class_to_idx = {c: i for i, c in enumerate(sorted(overlay_names))}
 
        overlay_col_by_class = {c: f"overlay_{c}" for c in class_to_idx}
        missing = [
            overlay_col_by_class[c] for c in class_to_idx
            if overlay_col_by_class[c] not in toc
        ]
        if missing:
            raise ValueError(
                f"{split}_toc.json is missing overlay column(s) seen in "
                f"train: {missing}"
            )
 
        # ---- base class ----
        if base_col is not None:
            raw_base = toc[base_col]
            if base_to_idx is None:
                base_classes = sorted(set(raw_base))
                base_to_idx = {c: i for i, c in enumerate(base_classes)}
            unseen = set(raw_base) - set(base_to_idx)
            if unseen:
                raise ValueError(
                    f"{split}_toc.json base column '{base_col}' has class(es) "
                    f"not seen in train: {unseen}"
                )
            base_indices = [base_to_idx[b] for b in raw_base]
        else:
            base_to_idx = {}
            base_indices = [None] * n
 
        labels = []
        for i in range(n):
            overlays = {
                c: int(toc[overlay_col_by_class[c]][i]) for c in class_to_idx
            }
            labels.append({"base": base_indices[i], "overlays": overlays})
 
    files = [os.path.join(data_dir, split, fn) for fn in file_names]
    return files, labels, class_to_idx, base_to_idx, label_format


# ----------------- base -> overlay reconciliation -----------------
def reconcile_base_overlay_labels(labels, base_to_idx, class_to_idx, strict=False):
    """
    For each "base_overlay"-format label dict, ensure the overlay entry
    matching the row's own base class is set to 1.

    The base audio's own class isn't necessarily reflected in the overlay
    flags (those typically mark sounds mixed IN on top of the base), so a
    clip whose base_audio is "Drones" should also have overlays["Drones"] == 1
    even if no separate drone sound was overlaid.

    Matching is done by class NAME: base_to_idx and class_to_idx are both
    name -> index maps, so we look up the base class's name and set the
    overlay of the same name, if one exists.

    Parameters
    ----------
    labels : list of dict
        As returned by load_toc for "base_overlay" format:
        [{"base": int or None, "overlays": {class_name: 0/1, ...}}, ...]
        Modified in place and also returned.
    base_to_idx : dict[str, int]
        Base class name -> index (from load_toc).
    class_to_idx : dict[str, int]
        Overlay class name -> index (from load_toc).
    strict : bool
        If True, raise an error when a base class has no matching overlay
        class name (instead of just leaving that row's overlays unchanged).

    Returns
    -------
    labels : list of dict (same list, mutated)
    """
    idx_to_base = {v: k for k, v in base_to_idx.items()}
    overlay_class_names = set(class_to_idx.keys())

    unmatched_base_classes = set(idx_to_base.values()) - overlay_class_names
    if unmatched_base_classes:
        msg = (
            f"base class(es) with no matching overlay column by name: "
            f"{unmatched_base_classes}. Rows with these base classes will be "
            f"left unchanged."
        )
        if strict:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    for label in labels:
        base_idx = label.get("base")
        if base_idx is None:
            continue
        base_name = idx_to_base.get(base_idx)
        if base_name in label["overlays"]:
            label["overlays"][base_name] = 1

    return labels


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

def add_tonality_and_bpf_features(df, n_fft=2048, hop_length=512, sr=None,
                                   filepath_col="filepath"):
    """
    Given an existing feature DataFrame (must contain a filepath column),
    load each audio file, compute low-band tonality/flatness and blade-pass
    comb-strength features, and return a new DataFrame with those columns
    merged in (aligned by row order / filepath).

    Does NOT recompute any of the original features — only adds the new ones.
    """
    new_rows = []
    n = len(df)

    for i, path in enumerate(df[filepath_col]):
        logger.info(f"Adding tonality/BPF features [{i + 1}/{n}]: {path}")
        try:
            y, sr_i = librosa.load(path, sr=sr, mono=True)
            S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

            feats = {}
            feats.update(extract_tonality_features(S, sr_i, n_fft))
            feats.update(extract_bpf_comb_features(S, sr_i, n_fft))
            feats[filepath_col] = path
            new_rows.append(feats)
        except Exception as e:
            logger.error(f"Failed to extract tonality/BPF features for {path}: {e}")
            # keep row alignment even on failure, fill with NaN
            new_rows.append({filepath_col: path})

    new_feats_df = pd.DataFrame(new_rows)

    # merge on filepath so row order/failures can't silently misalign columns
    updated_df = df.merge(new_feats_df, on=filepath_col, how="left")
    return updated_df


def extract_tonality_features(S, sr, n_fft, low_hz=50, high_hz=500):
    """Spectral flatness and peak-to-average ratio restricted to the
    low-frequency band, to separate narrowband rotor/motor tones
    (drones) from broadband engine/road noise (land vehicles)."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band_mask = (freqs >= low_hz) & (freqs < high_hz)
    band_power = (S[band_mask, :]) ** 2 + 1e-12

    log_power = np.log(band_power)
    geo_mean = np.exp(np.mean(log_power, axis=0))
    arith_mean = np.mean(band_power, axis=0)
    flatness = geo_mean / (arith_mean + 1e-12)

    peak = np.max(band_power, axis=0)
    par = peak / (arith_mean + 1e-12)

    feats = {}
    feats.update(summarize(flatness, "lowband_flatness"))
    feats.update(summarize(par, "lowband_peak_avg_ratio"))
    return feats


def extract_bpf_comb_features(S, sr, n_fft, search_low=30, search_high=250):
    """Autocorrelate the log-magnitude spectrum's low-frequency region
    to detect evenly-spaced harmonic combs characteristic of rotor
    blade-pass frequencies. High comb strength -> drone; vehicles
    (broadband rumble) score low."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= search_low) & (freqs <= search_high)
    mean_spec = np.mean(S[mask, :], axis=1)
    mean_spec = mean_spec - np.mean(mean_spec)

    autocorr = np.correlate(mean_spec, mean_spec, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)

    comb_strength = float(np.max(autocorr[3:])) if len(autocorr) > 3 else 0.0
    comb_lag_idx = int(np.argmax(autocorr[3:]) + 3) if len(autocorr) > 3 else 0

    return {
        "bpf_comb_strength": comb_strength,
        "bpf_comb_lag_bins": float(comb_lag_idx),
    }


# ----------------- DataFrame builder -----------------
def build_feature_table(data_dir, split, n_fft=2048, hop_length=512, sr=None,
                         n_mfcc=13, n_mels=128, class_to_idx=None, base_to_idx=None,
                         reconcile_base_labels=True, strict_reconcile=False):
    """
    Build a pandas DataFrame of summary features, one row per audio file.
 
    Parameters
    ----------
    split: str "train" or "test"
    n_fft, hop_length : int
        STFT parameters shared across all spectral/band features.
    sr : int, optional
        Force a target sample rate; None keeps each file's native rate.
    n_mfcc, n_mels : int
        MFCC coefficient count and mel filterbank size.
    class_to_idx : dict, optional
        Label/overlay-class encoding. Pass the value returned from the train
        call when building the test table, so encodings stay consistent
        across splits.
    base_to_idx : dict, optional
        Base-class encoding (only relevant for the base+overlay TOC format).
        Same idea: pass the train-split value in when building test.
    reconcile_base_labels : bool
        If True (default) and the TOC is "base_overlay" format, automatically
        set overlays[base_class_name] = 1 for each row, so the base audio's
        own class is reflected in the overlay flags even if it wasn't
        explicitly marked as an overlay. Matching is by class name between
        base_to_idx and class_to_idx.
    strict_reconcile : bool
        If True, raise an error when a base class name has no matching
        overlay class (instead of just warning and leaving those rows as-is).
 
    Returns
    -------
    df : pd.DataFrame
        - Original single-label TOCs: a 'label' column (int class index).
        - Base+overlay TOCs: a 'label_base' column (int index of primary
          source, or NaN if no base column existed) plus one
          'label_overlay_<class>' column per overlay class (0/1).
    class_to_idx : dict
        The encoding used — reuse this for the paired split.
    base_to_idx : dict
        The base-class encoding used — reuse this for the paired split.
    """
    audio_files, labels, class_to_idx, base_to_idx, label_format = load_toc(
        data_dir, split, class_to_idx=class_to_idx, base_to_idx=base_to_idx
    )
    logger.info(f"Detected TOC label format: {label_format}")
    if class_to_idx:
        logger.info(f"Class encoding: {class_to_idx}")
    if base_to_idx:
        logger.info(f"Base class encoding: {base_to_idx}")

    if label_format == "base_overlay" and reconcile_base_labels and base_to_idx:
        labels = reconcile_base_overlay_labels(
            labels, base_to_idx, class_to_idx, strict=strict_reconcile
        )
 
    if labels is not None and len(labels) != len(audio_files):
        raise ValueError("labels must be the same length as audio_files")
 
    rows = []
    for i, path in enumerate(audio_files):
        logger.info(f"Extracting features [{i + 1}/{len(audio_files)}]: {path}")
        try:
            row = extract_all_features(path, n_fft=n_fft, hop_length=hop_length, sr=sr,
                                        n_mfcc=n_mfcc, n_mels=n_mels)
            row["filepath"] = path
 
            if label_format == "single":
                row["label"] = labels[i]
            else:
                row["label_base"] = labels[i]["base"]
                for cls, val in labels[i]["overlays"].items():
                    row[f"label_overlay_{cls}"] = val
 
            rows.append(row)
        except Exception as e:
            logger.error(f"Failed to extract features for {path}: {e}")
 
    df = pd.DataFrame(rows)
 
    # keep filepath/label(s) as leading columns if present
    label_cols = [
        c for c in df.columns
        if c == "label" or c == "label_base" or c.startswith("label_overlay_")
    ]
    leading = (["filepath"] if "filepath" in df.columns else []) + label_cols
    other = [c for c in df.columns if c not in leading]
    df = df[leading + other]

    df = add_tonality_and_bpf_features(df)
 
    return df, class_to_idx, base_to_idx

        

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