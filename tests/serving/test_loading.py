from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.loading import ModelContractError, load_snapshot, validate_pipeline
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE


def test_local_round_trip(settings: ServingSettings, pipeline: Pipeline) -> None:
    model = load_snapshot(settings)
    frame = pd.DataFrame([APPLICANT_EXAMPLE["features"]])
    np.testing.assert_array_equal(
        model.predictor.predict_proba(frame), pipeline.predict_proba(frame)
    )
    assert model.features.names == tuple(frame.columns)
    assert model.features.nullable == frozenset(frame.columns)
    assert model.default_column == 1


def test_registry_exact_champion(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = Mock()
    registry.resolve_alias.return_value = "17"
    registry.load_pipeline.return_value = pipeline
    factory = Mock(return_value=registry)
    monkeypatch.setattr("credit_scoring.serving.loading.MLflowRegistry", factory)
    settings = ServingSettings(
        CREDIT_SCORING_MLFLOW_TRACKING_URI="http://registry",
        CREDIT_SCORING_REGISTERED_MODEL_NAME="shared-name",
    )
    load_snapshot(settings)
    factory.assert_called_once_with("http://registry", "credit-scoring", "shared-name")
    registry.resolve_alias.assert_called_once_with("champion")
    registry.load_pipeline.assert_called_once_with("17")


@pytest.mark.parametrize("failure", [None, RuntimeError("private registry failure")])
def test_registry_failure_has_no_local_fallback(
    failure: Exception | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Mock()
    registry.resolve_alias.return_value = None
    registry.resolve_alias.side_effect = failure
    local = Mock(side_effect=AssertionError("must not fall back"))
    monkeypatch.setattr("mlflow.sklearn.load_model", local)
    settings = ServingSettings(CREDIT_SCORING_MLFLOW_TRACKING_URI="http://registry")
    with pytest.raises((ModelContractError, RuntimeError)):
        load_snapshot(settings, registry=registry)
    registry.load_pipeline.assert_not_called()
    local.assert_not_called()


def test_local_failure_has_no_registry_fallback(
    settings: ServingSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = Mock()
    monkeypatch.setattr("mlflow.sklearn.load_model", Mock(side_effect=RuntimeError("broken")))
    with pytest.raises(RuntimeError):
        load_snapshot(settings, registry=registry)
    registry.resolve_alias.assert_not_called()


@pytest.mark.parametrize(
    "metadata",
    [
        {"feature_names_in_": None},
        {"feature_names_in_": []},
        {"feature_names_in_": [["AGE"]]},
        {"feature_names_in_": ["AGE", "AGE"]},
        {"feature_names_in_": ["ID"]},
        {"feature_names_in_": ["default.payment.next.month"]},
        {"feature_names_in_": [3]},
        {"predict_proba": None},
        {"classes_": None},
        {"classes_": [0]},
        {"classes_": [0, 1, 2]},
        {"classes_": [0, 2]},
        {"classes_": [1, 1]},
        {"classes_": ["0", "1"]},
        {"classes_": [False, True]},
        {"classes_": [0, np.nan]},
    ],
)
def test_incompatible_metadata(metadata: dict[str, object]) -> None:
    model = SimpleNamespace(
        **{"feature_names_in_": ["AGE"], "classes_": [0, 1], "predict_proba": Mock(), **metadata}
    )
    with pytest.raises(ModelContractError):
        validate_pipeline(model)


def test_unsupported_model_without_explanation() -> None:
    model = SimpleNamespace(feature_names_in_=["AGE"], classes_=[1, 0], predict_proba=Mock())
    with pytest.raises(ModelContractError, match="explanation"):
        validate_pipeline(model)
