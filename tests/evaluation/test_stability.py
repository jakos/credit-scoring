import numpy as np
import pandas as pd
import pytest

from credit_scoring.evaluation.metrics import evaluate_predictions
from credit_scoring.evaluation.stability import population_stability_index as psi


def test_discrimination_with_ties() -> None:
    metrics = evaluate_predictions(
        pd.Series([0, 1, 0, 1]), pd.Series([0, 0, 1, 1]), pd.Series([0.2, 0.2, 0.8, 0.8])
    )
    assert metrics["roc_auc"] == 0.5
    assert metrics["pr_auc"] == 0.5
    assert metrics["ks"] == 0
    assert metrics["gini"] == 0


def test_psi_expected_calculation_and_empty_bins() -> None:
    reference = [0.1, 0.2, 0.8, 0.9]
    assert psi(reference, reference, bins=2, smoothing=0.01) == 0
    expected = np.array([0.51, 0.51]) / 1.02
    actual = np.array([0.01, 1.01]) / 1.02
    assert psi(reference, [0.8, 0.9], bins=2, smoothing=0.01) == pytest.approx(
        sum((actual - expected) * np.log(actual / expected))
    )
    assert psi([0.5] * 4, [0.5] * 2, bins=10, smoothing=0.001) == 0
    assert psi([0.5] * 4, [0.1] * 4, bins=10, smoothing=0.001) > 1
    assert np.isfinite(psi([0, 0, 1, 1], [0, 1], bins=10, smoothing=0.001))


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-0.1], [1.1], [[0.1]]])
def test_invalid_scores(values) -> None:
    with pytest.raises(ValueError):
        psi(values, [0.5], bins=10, smoothing=0.001)
    with pytest.raises(ValueError):
        psi([0.5], values, bins=10, smoothing=0.001)


@pytest.mark.parametrize("bins,smoothing", [(1, 0.01), (True, 0.01), (2, 0), (2, float("inf"))])
def test_invalid_binning_parameters(bins, smoothing) -> None:
    with pytest.raises(ValueError):
        psi([0.1], [0.2], bins=bins, smoothing=smoothing)
