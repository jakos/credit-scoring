import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_scoring.modeling.evaluate import evaluate_predictions


def test_evaluate_predictions_returns_expected_metric_values() -> None:
    y_true = pd.Series([0, 1, 0, 1, 1])
    y_pred = pd.Series([0, 1, 0, 0, 1])
    y_proba = pd.Series([0.10, 0.80, 0.20, 0.40, 0.95])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics["precision"] == pytest.approx(precision_score(y_true, y_pred, zero_division=0))
    assert metrics["recall"] == pytest.approx(recall_score(y_true, y_pred, zero_division=0))
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred, zero_division=0))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_proba))
    assert metrics["pr_auc"] == pytest.approx(average_precision_score(y_true, y_proba))


def test_evaluate_predictions_precision_recall_f1_use_zero_division() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = pd.Series([0, 0, 0, 0])
    y_proba = pd.Series([0.05, 0.10, 0.20, 0.30])

    metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_predictions_returns_nan_roc_auc_for_single_class_ground_truth() -> None:
    y_true = pd.Series([0, 0, 0, 0])
    y_pred = pd.Series([0, 0, 0, 0])
    y_proba = pd.Series([0.1, 0.2, 0.3, 0.4])

    with pytest.warns(UndefinedMetricWarning, match="Only one class is present in y_true"):
        with pytest.warns(UserWarning, match="No positive class found in y_true"):
            metrics = evaluate_predictions(y_true, y_pred, y_proba)

    assert metrics["pr_auc"] == 0.0
    assert metrics["roc_auc"] == pytest.approx(float("nan"), nan_ok=True)
