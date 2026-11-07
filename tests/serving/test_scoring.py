from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from credit_scoring.serving.contracts import (
    Explanation,
    FeatureContract,
    FeatureContribution,
    ModelSnapshot,
)
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE
from credit_scoring.serving.scoring import FeatureValidationError, PredictionError, Scorer


def test_real_probabilities_order_and_vectorization(snapshot: ModelSnapshot) -> None:
    first = APPLICANT_EXAMPLE["features"]
    second = {**first, "AGE": 70, "LIMIT_BAL": None}
    shuffled = dict(reversed(list(first.items())))
    predictor = Mock(wraps=snapshot.predictor)
    scorer = Scorer(
        ModelSnapshot(predictor, snapshot.features, snapshot.default_column, snapshot.explainer),
        0.2,
        0.5,
    )
    results = scorer.score([first, second], batch=True)
    predictor.predict_proba.assert_called_once()
    frame = predictor.predict_proba.call_args.args[0]
    assert list(frame.columns) == list(snapshot.features.names)
    expected = snapshot.predictor.predict_proba(frame)[:, snapshot.default_column]
    np.testing.assert_array_equal([r.probability_of_default for r in results], expected)
    assert results == [scorer.score([first])[0], scorer.score([second])[0]]
    assert scorer.score([first]) == scorer.score([shuffled])


@pytest.mark.parametrize("medium,high", [(0.2, 0.5), (0.3, 0.8)])
def test_threshold_boundaries_and_reversed_class_column(medium: float, high: float) -> None:
    probabilities = [
        0,
        np.nextafter(medium, 0),
        medium,
        np.nextafter(medium, 1),
        np.nextafter(high, 0),
        high,
        np.nextafter(high, 1),
        1,
    ]
    predictor = SimpleNamespace(
        feature_names_in_=["AGE"],
        classes_=[1, 0],
        predict_proba=Mock(
            return_value=np.column_stack([probabilities, 1 - np.array(probabilities)])
        ),
    )
    explanation = Mock()
    explanation.explain.side_effect = lambda frame: [
        Explanation(
            method="linear_log_odds",
            baseline_log_odds=0,
            top_features=[FeatureContribution(feature="AGE", contribution=0)],
            remaining_contribution=0,
        )
        for _ in range(len(frame))
    ]
    scorer = Scorer(
        ModelSnapshot(predictor, FeatureContract(("AGE",)), 0, explanation), medium, high
    )
    results = scorer.score([{"AGE": 35}] * len(probabilities), batch=True)
    assert [r.probability_of_default for r in results] == probabilities
    assert [r.risk_band for r in results] == [
        "low",
        "low",
        "medium",
        "medium",
        "medium",
        "high",
        "high",
        "high",
    ]
    predictor.predict_proba.assert_called_once()


@pytest.mark.parametrize(
    "output",
    [
        [],
        [0.5, 0.5],
        [[0.5]],
        [[0.2, 0.3, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
        [[float("nan"), 0.5]],
        [[0.5, float("inf")]],
        [[0.5, -0.1]],
        [[1.1, 0.5]],
        [["bad", 0.5]],
    ],
)
def test_invalid_prediction_output(output: object) -> None:
    predictor = Mock()
    predictor.predict_proba.return_value = output
    scorer = Scorer(ModelSnapshot(predictor, FeatureContract(("AGE",)), 1), 0.2, 0.5)
    with pytest.raises(PredictionError):
        scorer.score([{"AGE": 35}])


def test_prediction_failure_is_distinct_and_sanitized() -> None:
    predictor = Mock()
    predictor.predict_proba.side_effect = RuntimeError("secret private path")
    scorer = Scorer(ModelSnapshot(predictor, FeatureContract(("AGE",)), 1), 0.2, 0.5)
    with pytest.raises(PredictionError, match=r"^Model prediction failed$") as caught:
        scorer.score([{"AGE": 35}])
    assert caught.value.__cause__ is predictor.predict_proba.side_effect


def test_feature_validation_precedes_prediction_with_batch_locations() -> None:
    predictor = Mock()
    scorer = Scorer(ModelSnapshot(predictor, FeatureContract(("AGE",)), 1), 0.2, 0.5)
    with pytest.raises(FeatureValidationError) as caught:
        scorer.score([{"AGE": 35}, {"secret": 99}], batch=True)
    assert caught.value.details[0].loc == ["body", "applicants", 1, "features", "AGE"]
    assert caught.value.details[1].type == "extra_forbidden"
    assert "secret" not in str(caught.value.details)
    predictor.predict_proba.assert_not_called()
    with pytest.raises(FeatureValidationError) as null_error:
        scorer.score([{"AGE": None}])
    assert null_error.value.details[0].type == "null_not_supported"


@pytest.mark.parametrize(
    "medium, high", [(0, 0.5), (0.5, 0.5), (0.6, 0.5), (0.2, 1), (float("nan"), 0.5)]
)
def test_invalid_injected_thresholds(snapshot: ModelSnapshot, medium: float, high: float) -> None:
    with pytest.raises(ValueError):
        Scorer(snapshot, medium, high)


@pytest.mark.parametrize("batch", [False, True])
def test_unknown_keys_have_distinct_insertion_positions(batch: bool) -> None:
    predictor = Mock()
    scorer = Scorer(ModelSnapshot(predictor, FeatureContract(("AGE",)), 1), 0.2, 0.5)
    with pytest.raises(FeatureValidationError) as caught:
        scorer.score([{"unknown-z-sentinel": 91, "AGE": 35, "unknown-a-sentinel": 92}], batch=batch)
    prefix = ["body", "applicants", 0] if batch else ["body"]
    assert [d.loc for d in caught.value.details] == [
        [*prefix, "features", 0],
        [*prefix, "features", 2],
    ]
    assert "sentinel" not in str(caught.value.details)
    predictor.predict_proba.assert_not_called()


@pytest.mark.parametrize("value", [1e308, -1e308, 10**400, -(10**400), 1e12 + 1, -1e12 - 1])
def test_direct_scoring_rejects_out_of_range_before_prediction(value: int | float) -> None:
    predictor = Mock()
    scorer = Scorer(ModelSnapshot(predictor, FeatureContract(("AGE",)), 1), 0.2, 0.5)
    with pytest.raises(FeatureValidationError) as caught:
        scorer.score([{"AGE": value}])
    assert caught.value.details[0].type == "feature_range"
    predictor.predict_proba.assert_not_called()


@pytest.mark.parametrize("value", [-1e12, 1e12, -(10**12), 10**12])
def test_direct_scoring_accepts_inclusive_bounds(
    snapshot: ModelSnapshot, value: int | float
) -> None:
    features = {**APPLICANT_EXAMPLE["features"], "AGE": value}
    assert len(Scorer(snapshot, 0.2, 0.5).score([features])) == 1
