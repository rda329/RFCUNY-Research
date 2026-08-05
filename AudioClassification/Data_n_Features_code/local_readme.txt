================================================================================
 AUDIO EVENT CLASSIFICATION — LOCAL README
 (gunshot / explosion / drone / other)
================================================================================

PIPELINE
--------
1. data.py         -> bootstrap/augment raw audio, write a TOC json
2. features_df.py  -> read TOC, extract features, build a DataFrame
3. Train a model (Random Forest / XGBoost) on the DataFrame
4. build_feature_row_for_inference() to score new files

Deps: librosa, numpy, pandas, soundfile
  pip install librosa numpy pandas soundfile


================================================================================
 data.py — AudioBootstrapper
================================================================================

BootStrap_v1(og_folder_path, sub_folder_names, output_folder_path,
             aug_technique_num, num_bootstraps, toc_name)
  - Single-label augmentation only (no overlay mixing).
  - 4 techniques (1-4 applied per file): time stretch, pitch shift,
    gaussian noise, time shift.
  - Writes {toc_name}.json:  {"file_name": [...], "label": [...]}

BootStrap_v2(..., allow_self_overlay=True)
  - Same as v1 + a 5th technique: OVERLAY (mixes in audio from any class).
  - Base clip padded to >=60s before overlaying.
  - Randomly picks 0+ overlay classes (excludes own class unless
    allow_self_overlay=False, with fallback if no others exist).
  - Writes {toc_name}.json:
        {"file_name": [...], "base_audio": [...],
         "overlay_<Class>": [0/1, ...]}   (one column per class)

Output paths:
  - Audio  -> output_folder_path/audio_<n>.wav
  - TOC    -> output_folder_path.parent/{toc_name}.json   (one dir up!)

Note: a clip's own base class is NOT auto-marked as an overlay by data.py
— features_df.py fixes that at read time (see reconcile below).


================================================================================
 features_df.py — Feature Extraction
================================================================================

load_toc(data_dir, split, class_to_idx=None, base_to_idx=None)
  Auto-detects TOC format from data/{split}_toc.json:
    "single"       -> has "label" key
    "base_overlay" -> binary columns = overlay classes, the one
                       remaining non-binary column = base class
  Pass train's class_to_idx/base_to_idx in when loading test, so
  encodings stay consistent across splits.

reconcile_base_overlay_labels(labels, base_to_idx, class_to_idx)
  For base_overlay TOCs: sets overlays[<own base class>] = 1 for each
  row (matched by NAME). Runs automatically inside build_feature_table
  unless reconcile_base_labels=False.

Feature groups extracted per file:
  - MFCC / delta / delta2        (timbre)
  - Spectral centroid/bandwidth/rolloff/flatness/flux
  - Temporal: ZCR, RMS, attack/decay time, temporal centroid, crest factor
  - Harmonic: F0 mean/std, voiced fraction, HNR
  - Band energy ratios: low/mid/high + rotor-fundamental (80-300Hz)
  - Tonality (low-band flatness/peak-avg ratio) + BPF comb strength
    (detects drone rotor blade-pass frequency combs)
  See the "Feature Cheat Sheet" docstring at the bottom of features_df.py
  for what each one means / how to read high vs low values.

build_feature_table(data_dir, split, ..., class_to_idx=None,
                     base_to_idx=None, reconcile_base_labels=True)
  Main entry point. Returns:
    df            -> one row per file; leading cols = filepath + label(s)
                        single:        "label"
                        base_overlay:  "label_base", "label_overlay_<class>"
    class_to_idx, base_to_idx -> reuse these for the paired split
  Files that fail extraction are silently dropped (logged).

build_feature_row_for_inference(path, feature_columns=None, ...)
  Extracts the full feature set for ONE file -> single-row DataFrame.
  Pass feature_columns (train df's non-label columns) to reindex to the
  exact training column order — required for sklearn/XGBoost predict().


================================================================================
 GOTCHAS
================================================================================
- File layout expected: {data_dir}/{split}_toc.json + audio at
  {data_dir}/{split}/{file_name}. Match this to data.py's output paths.
- Always reuse train's class_to_idx/base_to_idx when building test.
- Base/overlay reconciliation matches by name — mismatched names get
  skipped (warned), or use strict_reconcile=True to raise instead.
- Every audio file is loaded TWICE (once for core features, again for
  tonality/BPF features) — known inefficiency, not a bug.
================================================================================