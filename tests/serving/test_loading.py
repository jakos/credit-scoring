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


def test_real_registry_construction_and_loading_are_read_only(
    monkeypatch: pytest.MonkeyPatch, pipeline: Pipeline
) -> None:
    import credit_scoring.registry.mlflow_adapter as adapter

    forbidden = Mock(side_effect=AssertionError("write forbidden"))
    client = SimpleNamespace(
        get_registered_model=Mock(return_value=SimpleNamespace(aliases={"champion": "7"})),
        set_registered_model_alias=forbidden,
        set_model_version_tag=forbidden,
    )
    mlflow = SimpleNamespace(
        set_tracking_uri=Mock(),
        set_experiment=forbidden,
        start_run=forbidden,
        register_model=forbidden,
        log_text=forbidden,
        set_tags=forbidden,
        tracking=SimpleNamespace(MlflowClient=Mock(return_value=client)),
        sklearn=SimpleNamespace(load_model=Mock(return_value=pipeline)),
    )
    monkeypatch.setattr(adapter, "import_module", Mock(return_value=mlflow))
    configuration = ServingSettings(CREDIT_SCORING_MLFLOW_TRACKING_URI="http://registry")
    snapshot = load_snapshot(configuration)
    assert snapshot.predictor is pipeline
    client.get_registered_model.assert_called_once_with("credit-scoring")
    mlflow.sklearn.load_model.assert_called_once_with("models:/credit-scoring/7")
    forbidden.assert_not_called()


def test_loading_wrappers_preserve_causes(monkeypatch: pytest.MonkeyPatch) -> None:
    from credit_scoring.serving.loading import RegistryLoadError

    cause = ConnectionError("private-tracking-uri")
    registry = Mock()
    registry.resolve_alias.side_effect = cause
    settings = ServingSettings(CREDIT_SCORING_MLFLOW_TRACKING_URI="http://registry")
    with pytest.raises(RegistryLoadError) as caught:
        load_snapshot(settings, registry=registry)
    assert caught.value.__cause__ is cause
    cause = ValueError("private-artifact-details")
    monkeypatch.setattr("credit_scoring.serving.loading.build_explanation", Mock(side_effect=cause))
    model = SimpleNamespace(feature_names_in_=["AGE"], classes_=[0, 1], predict_proba=Mock())
    with pytest.raises(ModelContractError, match="explanation") as caught:
        validate_pipeline(model)
    assert caught.value.__cause__ is cause
