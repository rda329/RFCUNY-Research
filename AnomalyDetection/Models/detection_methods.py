""" 
This module will implement a pipeline for the purpose of detecting 
anomalies in the audio files through the use of primary component analysis (PCA)
"""
from sklearn.decomposition import PCA
import logging
import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from ..Data_n_Features_code.DataManipulate import DataChef
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score, precision_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt




#Print statements system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AnomalyDetection_methods():
    def __init__(self,):
        pass

    def PCA_pipeline(self, natural_bool: bool, anomaly_type: str, window_size: int) -> pd.DataFrame:
        logger.info(f"Starting PCA Method pipeline | natural={natural_bool}, anomaly_type={anomaly_type}, window_size={window_size}")

        chef = DataChef()
        audio_dict = chef.GetAudios(natural_bool, anomaly_type)
        logger.info(f"Loaded {len(audio_dict)} audio files")

        # For averaged curves
        mean_fpr    = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)
        roc_tprs, roc_aucs   = [], []
        pr_precs,  pr_aucs   = [], []

        counter = 0
        for key, val in audio_dict.items():
            df = chef.PrepData_Method(val, window_size)

            target_var = 0.99 #target variance that we aim to obtain by n number of components
            mean_reconstruction_error = self._PCA_method(target_var, df)
            df["mean_reconstruction_error"] = mean_reconstruction_error

            # --- ROC curve ---
            fpr, tpr, _ = roc_curve(df["anomaly_bool"], df["mean_reconstruction_error"])
            roc_tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_aucs.append(auc(fpr, tpr))

            # --- PR curve ---
            prec, rec, _ = precision_recall_curve(df["anomaly_bool"], df["mean_reconstruction_error"])
            # sklearn returns descending recall; flip so recall is ascending for interp
            pr_precs.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
            pr_aucs.append(average_precision_score(df["anomaly_bool"], df["mean_reconstruction_error"]))

            counter += 1
            logger.debug(f"Progress: {counter}/{len(audio_dict)}")

        # --- Plot ---
        self._plot_mean_curves(
            mean_fpr, roc_tprs, roc_aucs,
            mean_recall, pr_precs, pr_aucs
        )

        logger.info("Pipeline complete.")

    #Main method to get evaluation results for Isolation Forest Method
    def IsolationForest_pipeline(self, natural_bool: bool, anomaly_type: str, window_size: int, test_ratio: float, contamination_param=0.01) -> pd.DataFrame:
        logger.info(f"Starting IsolationForest pipeline | natural={natural_bool}, anomaly_type={anomaly_type}, window_size={window_size}, test_ratio={test_ratio}, contamination={contamination_param}")

        chef = DataChef()
        audio_dict = chef.GetAudios(natural_bool, anomaly_type)
        logger.info(f"Loaded {len(audio_dict)} audio files")

        precision_list, recall_list = [], []
        tn_list, fp_list, fn_list, tp_list = [], [], [], []

        # For averaged curves
        mean_fpr    = np.linspace(0, 1, 100)
        mean_recall = np.linspace(0, 1, 100)
        roc_tprs, roc_aucs   = [], []
        pr_precs,  pr_aucs   = [], []

        counter = 0
        for key, val in audio_dict.items():
            df = chef.PrepData_Method(val, window_size)

            train_df, test_df = train_test_split(
                df,
                test_size=test_ratio,
                stratify=df["anomaly_bool"],
                random_state=42
            )
            logger.debug(f"    Train size={len(train_df)}, Test size={len(test_df)}")

            train_scores, train_labels, test_scores, test_labels = self._IsolationForest_method(
                train_df, test_df, contamination_param
            )

            precision, recall, tn, fp, fn, tp = self._GetMetrics(test_df["anomaly_bool"], test_labels)
            logger.debug(f"    [{key}] precision={precision:.3f}, recall={recall:.3f}, tn={tn}, fp={fp}, fn={fn}, tp={tp}")

            precision_list.append(precision)
            recall_list.append(recall)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            tp_list.append(tp)

            # --- ROC curve ---
            # IsolationForest scores are negative (more negative = more anomalous),
            # so negate them so higher score = more likely anomaly
            fpr, tpr, _ = roc_curve(test_df["anomaly_bool"], -test_scores)
            roc_tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_aucs.append(auc(fpr, tpr))

            # --- PR curve ---
            prec, rec, _ = precision_recall_curve(test_df["anomaly_bool"], -test_scores)
            # sklearn returns descending recall; flip so recall is ascending for interp
            pr_precs.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
            pr_aucs.append(average_precision_score(test_df["anomaly_bool"], -test_scores))

            counter += 1
            logger.debug(f"Progress: {counter}/{len(audio_dict)}")

        # --- Build metrics DataFrame ---
        metrics_df = pd.DataFrame({
            "precision": precision_list,
            "recall":    recall_list,
            "tn":        tn_list,
            "fp":        fp_list,
            "fn":        fn_list,
            "tp":        tp_list,
            "roc_auc":   roc_aucs,
            "pr_auc":    pr_aucs,
        })

        # --- Plot ---
        self._plot_mean_curves(
            mean_fpr, roc_tprs, roc_aucs,
            mean_recall, pr_precs, pr_aucs
        )

        logger.info("Pipeline complete.")
        audio_indices = np.arange(0,len(metrics_df))
        metrics_df["audio_index"] = audio_indices
        return metrics_df

    
#-------------------------- Helper Functions ------------------------------------
    #Visualize metrics
    def _plot_mean_curves(self, mean_fpr, roc_tprs, roc_aucs, mean_recall, pr_precs, pr_aucs):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- ROC ---
        mean_tpr = np.mean(roc_tprs, axis=0)
        std_tpr  = np.std(roc_tprs, axis=0)
        mean_tpr[0], mean_tpr[-1] = 0.0, 1.0  # anchor endpoints

        ax = axes[0]
        ax.plot(mean_fpr, mean_tpr, color="steelblue", lw=2,
                label=f"Mean ROC (AUC = {np.mean(roc_aucs):.3f} ± {np.std(roc_aucs):.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        alpha=0.2, color="steelblue", label="±1 SD")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
        ax.legend(loc="lower right")

        # --- PR ---
        mean_prec = np.mean(pr_precs, axis=0)
        std_prec  = np.std(pr_precs, axis=0)

        ax = axes[1]
        ax.plot(mean_recall, mean_prec, color="darkorange", lw=2,
                label=f"Mean PR (AP = {np.mean(pr_aucs):.3f} ± {np.std(pr_aucs):.3f})")
        ax.fill_between(mean_recall, mean_prec - std_prec, mean_prec + std_prec,
                        alpha=0.2, color="darkorange", label="±1 SD")
        ax.set(xlabel="Recall", ylabel="Precision", title="PR Curve")
        ax.legend(loc="upper right")

        plt.suptitle("Mean Curves Across Audio Files", fontsize=13)
        plt.tight_layout()
        plt.show()
        
    
    def _GetMetrics(self, y_true: pd.Series, y_pred: np.array) -> dict:
        """
        y_true: ground truth labels (0 = normal, 1 = anomaly)
        y_pred: predicted labels (0 = normal, 1 = anomaly)
        """
        y_true = y_true.to_numpy()

        recall    = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        cm        = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel()
        return precision, recall, tn, fp, fn, tp
        

    def _IsolationForest_method(self, train_df: pd.DataFrame, test_df: pd.DataFrame, model_contamination_param: float):
        """
        train_df, test_df: pd dataframe, assumes structure provided by DataChef class PrepareData_method
        model_contamination_param: IsolationForest expected ratio of anomalies range (0,1)
        threshold: classification threshold
        """
        #mini function to get df ready for model input
        def fix_df(df):
            df = df.drop(columns= ["segment_index", "anomaly_bool"])
            return df.to_numpy()

        # Prep df
        X_train = fix_df(train_df)
        X_test = fix_df(test_df)
        
        # Train Isolation Forest
        model = IForest(contamination=model_contamination_param)
        model.fit(X_train)

        # Train
        train_scores = model.decision_scores_
        train_labels = model.labels_

        # Test — use decision_scores_ convention by scoring manually
        test_scores = model.decision_function(X_test)
        test_labels = model.predict(X_test)

        """ 
        train_scores, test_score: anomaly scores for train test data
        train_labels, test_labels: labels after predicting post model fit
        """

        return train_scores, train_labels, test_scores, test_labels
    
    def _PCA_method(self, target_var: float, audio_df: pd.DataFrame,):
        """ 
        target_var: As a decimal, percentage variance you which to capture
        X_train: numpy vector of mfccs in frequency domain
        """
        #mini function to get df ready for model input
        def fix_df(df):
            df = df.drop(columns= ["segment_index", "anomaly_bool"])
            return df.to_numpy()

        X_train = fix_df(audio_df)

        #Determine components needed to retain target_var variance
        pca = PCA(n_components= target_var)
        pca.fit(X_train)
        logger.info(f"Components needed for 99% variance: {pca.n_components_}")

        # Transform and reconstruct
        X_train_pca = pca.transform(X_train)
        X_train_reconstructed = pca.inverse_transform(X_train_pca)

        # Calculate reconstruction error
        reconstruction_error = (X_train - X_train_reconstructed).mean(axis=1)
        
        return reconstruction_error



if __name__ == "__main__":
    pass