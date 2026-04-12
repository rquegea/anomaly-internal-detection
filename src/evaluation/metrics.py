"""
Standard evaluation metrics for anomaly detection.
ESA recommends F0.5 (FPs more costly than FNs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """
    Compute the full metric suite used in the OPS-SAT-AD benchmark.

    Parameters
    ----------
    y_true  : ground truth binary labels
    y_pred  : binary predictions
    y_score : anomaly scores (higher = more anomalous); used for AUC-ROC

    Returns
    -------
    dict with precision, recall, f1, f05, auc_roc
    """
    result = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "f05":       fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
    }
    if y_score is not None:
        result["auc_roc"] = roc_auc_score(y_true, y_score)
    return result


def metrics_table(results: dict[str, dict]) -> pd.DataFrame:
    """Convert a {model_name: metrics_dict} mapping to a formatted DataFrame."""
    return pd.DataFrame(results).T[["precision", "recall", "f1", "f05", "auc_roc"]].round(3)
