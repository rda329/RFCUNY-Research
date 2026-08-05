"""
Local demo server for the xgboost audio classifiers (single-label + multi-label).

Run:
    pip install flask
    python app.py
Then open http://127.0.0.1:5000 in a browser.

This must live in the same folder as Demo_Code.py / Data_n_Features_code.py
(or that folder must be on your PYTHONPATH), since it reuses your
feature-extraction function directly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from flask import Flask, request, jsonify, send_file, render_template

# app.py lives in AudioClassification/Demo_Code/, and Data_n_Features_code
# is a sibling package under AudioClassification/. Add that parent folder
# to sys.path so a plain absolute import works when running `python app.py`
# directly (relative imports like `from ..X import y` only work when the
# file is launched as part of a package with `python -m`, not as a script).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Data_n_Features_code import build_feature_row_for_inference

app = Flask(__name__)

# Cache loaded models by directory so repeat clicks don't reload from disk every time.
_model_cache = {}


def load_multilabel(model_dir: Path):
    key = ("multi", str(model_dir))
    if key in _model_cache:
        return _model_cache[key]

    clf = XGBClassifier()
    clf.load_model(model_dir / "multilabel_xgb_model.json")

    with open(model_dir / "feature_cols.json") as f:
        feature_columns = json.load(f)
    with open(model_dir / "target_names.json") as f:
        target_names = json.load(f)

    bundle = (clf, feature_columns, target_names)
    _model_cache[key] = bundle
    return bundle


def load_singlelabel(model_dir: Path):
    key = ("single", str(model_dir))
    if key in _model_cache:
        return _model_cache[key]

    bst = xgb.Booster()
    bst.load_model(model_dir / "singlelabel_xgb_model.json")

    with open(model_dir / "feature_cols.json") as f:
        feature_columns = json.load(f)
    with open(model_dir / "class_to_idx.json") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    bundle = (bst, feature_columns, idx_to_class)
    _model_cache[key] = bundle
    return bundle


@app.route("/")
def index():
    return render_template("index.html")


def run_multilabel(model_dir: Path, audio_path: Path) -> dict:
    clf, feature_columns, target_names = load_multilabel(model_dir)
    row_df = build_feature_row_for_inference(str(audio_path), feature_columns=feature_columns)
    X = row_df.drop(columns=["filepath"], errors="ignore")

    pred_proba = clf.predict_proba(X)[0]
    pred_binary = (pred_proba >= 0.5).astype(int)

    predictions = [
        {"name": name, "present": bool(p), "prob": float(pr)}
        for name, p, pr in zip(target_names, pred_binary, pred_proba)
    ]
    predictions.sort(key=lambda d: -d["prob"])
    return {"mode": "multi", "predictions": predictions}


def run_singlelabel(model_dir: Path, audio_path: Path) -> dict:
    bst, feature_columns, idx_to_class = load_singlelabel(model_dir)
    row_df = build_feature_row_for_inference(str(audio_path), feature_columns=feature_columns)
    X = row_df.drop(columns=["filepath"], errors="ignore")

    dX = xgb.DMatrix(X)
    pred_proba = bst.predict(dX)[0]
    pred_idx = int(np.argmax(pred_proba))

    predictions = [
        {"name": idx_to_class[i], "present": i == pred_idx, "prob": float(p)}
        for i, p in enumerate(pred_proba)
    ]
    predictions.sort(key=lambda d: -d["prob"])
    return {
        "mode": "single",
        "predictions": predictions,
        "predicted_class": idx_to_class[pred_idx],
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    multi_dir_raw = (data.get("multi_model_dir") or "").strip()
    single_dir_raw = (data.get("single_model_dir") or "").strip()
    audio_path_raw = (data.get("audio_path") or "").strip()

    if not audio_path_raw:
        return jsonify({"error": "Audio file path is required."}), 400
    if not multi_dir_raw and not single_dir_raw:
        return jsonify({"error": "At least one model directory (multi or single) is required."}), 400

    audio_path = Path(audio_path_raw)
    if not audio_path.exists():
        return jsonify({"error": f"Audio file not found: {audio_path}"}), 400

    response = {
        "audio_name": audio_path.name,
        "audio_url": f"/audio?path={audio_path}",
        "multi": None,
        "single": None,
    }

    if multi_dir_raw:
        multi_dir = Path(multi_dir_raw)
        if not multi_dir.exists():
            response["multi"] = {"error": f"Model directory not found: {multi_dir}"}
        else:
            try:
                response["multi"] = run_multilabel(multi_dir, audio_path)
            except Exception as e:
                response["multi"] = {"error": f"Multi-label inference failed: {e}"}

    if single_dir_raw:
        single_dir = Path(single_dir_raw)
        if not single_dir.exists():
            response["single"] = {"error": f"Model directory not found: {single_dir}"}
        else:
            try:
                response["single"] = run_singlelabel(single_dir, audio_path)
            except Exception as e:
                response["single"] = {"error": f"Single-label inference failed: {e}"}

    return jsonify(response)


@app.route("/audio")
def audio():
    path = request.args.get("path")
    if not path:
        return "missing path", 400
    p = Path(path)
    if not p.exists():
        return "not found", 404

    # Flask/Python's mimetype guesser doesn't always know .flac, and getting
    # the Content-Type wrong causes the browser to silently refuse playback.
    ext_to_mime = {
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }
    mimetype = ext_to_mime.get(p.suffix.lower())

    # conditional=True enables HTTP range requests, so the browser can
    # seek/scrub the player instead of only playing straight through.
    return send_file(p, mimetype=mimetype, conditional=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)