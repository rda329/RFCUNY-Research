""" 
This module contains the code for 
Xgboost single and multi-label classification.
"""

import json
from xgboost import XGBClassifier
import xgboost as xgb
from Data_n_Features_code import build_feature_row_for_inference
from pathlib import Path
from IPython.display import Audio, display
import numpy as np
import librosa
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#Multi-label xgboost classifier
class Demo_1():
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.loaded_clf = None
        self.feature_columns = None
        self.target_names = None
        self._load_stuff_in()

    def Guess_That_Audio(self, audio_path: Path) -> None:
        row_df = build_feature_row_for_inference(str(audio_path), feature_columns=self.feature_columns)

        # Feed features to model
        X = row_df.drop(columns=["filepath"], errors="ignore")
        pred_proba = self.loaded_clf.predict_proba(X)[0]     # shape (n_labels,) for native multi-label XGBClassifier
        pred_binary = (pred_proba >= 0.5).astype(int)

        # Display label predictions
        print(f"\nPredictions for: {audio_path.name}")
        for name, p, pr in zip(self.target_names, pred_binary, pred_proba):
            flag = "✅" if p == 1 else "—" #Model prediction check if in audio else -
            print(f"  {flag} {name:<20} prob={pr:.3f}")


        # Allow user to play audio sound
        y_playback, sr_playback = librosa.load(str(audio_path), sr=None, mono=True)
        display(Audio(y_playback, rate=sr_playback))


#------------ Helper Functions -----------------------
    #Loads model, label names, and feature_col names
    def _load_stuff_in(self,) ->  None: 
        self.loaded_clf = XGBClassifier()
        self.loaded_clf.load_model(self.model_dir / "multilabel_xgb_model.json")

        # ordered list of feature col names used in training
        with open(self.model_dir / "feature_cols.json") as f:
            self.feature_columns = json.load(f)          

        with open(self.model_dir / "target_names.json") as f:
            self.target_names = json.load(f)
      

#Single label xgboost classifier
class Demo_2():
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.bst = None
        self.feature_columns = None
        self.class_to_idx = None
        self.idx_to_class = None
        self._load_stuff_in()

    def Guess_That_Audio(self,audio_path: Path) -> None:
        row_df = build_feature_row_for_inference(str(audio_path), feature_columns=self.feature_columns)
        X = row_df.drop(columns=["filepath"], errors="ignore")

        dX = xgb.DMatrix(X)
        pred_proba = self.bst.predict(dX)[0]      # shape (n_classes,)
        pred_idx = int(np.argmax(pred_proba))
        pred_class = self.idx_to_class[pred_idx]

        print(f"Predicted class: {pred_class}  (prob={pred_proba[pred_idx]:.3f})")

        # Allow user to play audio sound
        y_playback, sr_playback = librosa.load(str(audio_path), sr=None, mono=True)
        display(Audio(y_playback, rate=sr_playback))
    

#-------------- Helper Functions ------------
    def _load_stuff_in(self,) -> None:
        self.bst = xgb.Booster()
        self.bst.load_model(self.model_dir / "singlelabel_xgb_model.json")

        with open(self.model_dir / "feature_cols.json") as f:
            self.feature_columns = json.load(f)
        with open(self.model_dir / "class_to_idx.json") as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}


if __name__ == "__main__":
    pass