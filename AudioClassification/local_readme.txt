This directory explores audio classification through the lens of supervised learning.
The following models of used.
- Convolutional Neural Network: ./CNN
- Random Forest, Xgboost
    - Single Label Classification (v1)
    - Multi-Label Classification (v2)

================================================================================
 AUDIO EVENT CLASSIFICATION — LOCAL README
 (gunshot / explosion / drone / other)
================================================================================

PIPELINE
--------

1. Raw audio
   One folder per class (Explosion/, Gunshots/, UAV/, ...) under a common
   parent dir.

2. Bootstrap/augment  (data.py, AudioBootstrapper)
   Choose BootStrap_v1 (single-label, no overlay mixing) or BootStrap_v2
   (adds overlay mixing -> multi-label clips, e.g. drone + gunshot).
   Run once per split (toc_name="train_toc" / "test_toc").
   Outputs: audio_<n>.wav files + a {toc_name}.json TOC one dir above
   the audio folder. features_df.py expects these at
   {data_dir}/{split}_toc.json and {data_dir}/{split}/*.wav.

3. Build the train DataFrame  (features_df.py)
   build_feature_table(data_dir, "train") -> train_df, class_to_idx,
   base_to_idx. Auto-detects single vs. base_overlay TOC format,
   reconciles base/overlay labels by name (base_overlay only), extracts
   all features per file, drops files that fail. Keep class_to_idx /
   base_to_idx — you'll reuse them next.

4. Build the test DataFrame with the SAME encoding
   build_feature_table(data_dir, "test", class_to_idx=class_to_idx,
   base_to_idx=base_to_idx) -> test_df. Passing train's encodings in
   keeps label columns consistent across splits (raises if test has an
   unseen class).

5. Train / evaluate
   Split X/y (see local_readme_single_format_df.txt or
   local_readme_base_overlay_format_df.txt for exactly which columns to
   use — differs by TOC format). Fit Random Forest / XGBoost on train_df,
   evaluate on test_df.

6. Score a new file
   build_feature_row_for_inference(new_file_path,
   feature_columns=<train X's columns>) -> single-row DataFrame,
   column-aligned for .predict()/.predict_proba(). Map the predicted
   index back to a name via the inverse of class_to_idx / base_to_idx.

Deps: librosa, numpy, pandas, soundfile
  pip install librosa numpy pandas soundfile

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