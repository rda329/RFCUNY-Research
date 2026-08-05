# Data_n_Features Module

This directory contains utilities for generating synthetic audio datasets and preparing them for machine learning.

## Files

### `create_syn_audio.py`

Creates synthetic audio samples by overlaying anomaly sounds (e.g., explosions, gunshots, UAVs) onto background environmental recordings.

**Responsibilities**

* Maintains tables of contents (TOCs) for Urban and NonUrban datasets.
* Segments base audio into fixed-length clips.
* Randomly overlays anomalies onto a user-defined fraction of segments.
* Saves:

  * Synthetic `.wav` audio files
  * Corresponding `.json` metadata describing the injected anomalies

Output is organized as:

```text
Data/Audio_Files/Clips/
├── Urban/
│   └── audio_<id>/
│       └── <Anomaly_Type>/
│           ├── *.wav
│           └── *.json
└── NonUrban/
```

---

### `DataManipulate.py`

Processes the synthetic audio produced by `create_syn_audio.py` into a machine-learning-ready format.

**Responsibilities**

* Loads generated audio and metadata.
* Segments audio using the metadata.
* Applies preprocessing:

  * Moving-average denoising
  * FFT transformation to the frequency domain
  * Magnitude normalization
* Labels each segment as anomaly/non-anomaly.
* Returns a Pandas DataFrame where:

  * Each row represents one audio segment.
  * Columns contain FFT features.
  * `anomaly_bool` is the target label.

Also provides helper functions for locating generated audio files by environment (Urban/NonUrban) and anomaly type.

## Workflow

```text
Raw Audio
      │
      ▼
create_syn_audio.py
      │
      ▼
Synthetic Audio (.wav + .json)
      │
      ▼
DataManipulate.py
      │
      ▼
Processed Pandas DataFrame
      │
      ▼
Machine Learning Pipeline
```
