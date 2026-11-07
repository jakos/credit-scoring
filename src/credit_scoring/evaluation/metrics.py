"""Evaluation helpers for binary classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, float]:
    """Compute common classification metrics for model evaluation."""
    from credit_scoring.evaluation.stability import probabilities

    scores = probabilities(y_proba)
    auc = float(roc_auc_score(y_true, scores))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, scores)
        ks = float(np.max(np.abs(tpr - fpr)))
    else:
        ks = float("nan")
    return {
        "ks": ks,
        "gini": 2 * auc - 1,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }
