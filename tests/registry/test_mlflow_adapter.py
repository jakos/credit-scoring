from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from credit_scoring.registry.mlflow_adapter import MLflowRegistry


def adapter():
    mlflow = Mock()
    mlflow.start_run.return_value = nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="run")))
    mlflow.sklearn.log_model.return_value.model_uri = "models:/returned-id"
    mlflow.register_model.return_value.version = "17"
    client = Mock()
    client.get_model_version.return_value.status = "READY"
    return (
        MLflowRegistry("http://localhost:5000", "credit", "risk", mlflow=mlflow, client=client),
        mlflow,
        client,
    )


def test_exact_registration_alias_load_and_artifact_calls() -> None:
    registry, mlflow, client = adapter()
    pipeline = SimpleNamespace(named_steps={"model": Mock()})
    pipeline.named_steps["model"].get_params.return_value = {"max_iter": 50}
    model = registry.log_model(
        "lr",
        pipeline,
        {"roc_auc": 0.8},
        {"dataset_fingerprint": "abc"},
        lambda model: "pending card",
    )
    assert (model.version, model.run_id, model.model_uri) == ("17", "run", "models:/returned-id")
    mlflow.register_model.assert_called_once_with(
        "models:/returned-id",
        "risk",
        await_registration_for=300,
        tags={"algorithm": "lr", "dataset_fingerprint": "abc"},
    )
    mlflow.log_params.assert_called_once_with({"max_iter": 50})
    client.get_model_version.assert_called_once_with("risk", "17")
    registry.assign_alias("champion", "17")
    client.set_registered_model_alias.assert_called_once_with("risk", "champion", "17")
    client.get_registered_model.return_value.aliases = {"champion": 17}
    assert registry.resolve_alias("champion") == "17"
    client.get_registered_model.assert_called_once_with("risk")
    registry.load_pipeline("17")
    mlflow.sklearn.load_model.assert_called_once_with("models:/risk/17")
    registry.record(model, "# Model card", {"promotion_status": "promoted"})
    assert mlflow.log_text.call_args_list[0].args == ("pending card", "model_card.md")
    assert mlflow.log_text.call_args_list[-1].args == ("# Model card", "model_card.md")
    client.set_model_version_tag.assert_called_once_with(
        "risk", "17", "promotion_status", "promoted"
    )


@pytest.mark.parametrize(
    "code,missing",
    [
        ("RESOURCE_DOES_NOT_EXIST", True),
        ("PERMISSION_DENIED", False),
        ("INTERNAL_ERROR", False),
        (None, False),
    ],
)
def test_only_specific_not_found_bootstraps(code: str | None, missing: bool) -> None:
    registry, _, client = adapter()
    error = RuntimeError("registry failed")
    error.error_code = code
    client.get_registered_model.side_effect = error
    if missing:
        assert registry.resolve_alias("champion") is None
    else:
        with pytest.raises(RuntimeError, match="registry failed"):
            registry.resolve_alias("champion")


def test_existing_model_without_champion_bootstraps() -> None:
    registry, _, client = adapter()
    client.get_registered_model.return_value.aliases = {"challenger": "17"}

    assert registry.resolve_alias("champion") is None

    client.get_registered_model.assert_called_once_with("risk")
    client.get_model_version_by_alias.assert_not_called()


@pytest.mark.parametrize("operation", ["registration", "readiness", "loading", "alias", "artifact"])
def test_failures_propagate(operation) -> None:
    registry, mlflow, client = adapter()
    pipeline = SimpleNamespace(named_steps={"model": Mock()})
    if operation == "registration":
        mlflow.register_model.side_effect = RuntimeError("failed")
    if operation == "readiness":
        client.get_model_version.return_value.status = "PENDING_REGISTRATION"
    if operation in {"registration", "readiness"}:
        with pytest.raises(RuntimeError):
            registry.log_model("lr", pipeline, {}, {}, lambda model: "pending card")
        mlflow.log_text.assert_called_once_with("pending card", "model_card.md")
    elif operation == "loading":
        mlflow.sklearn.load_model.side_effect = RuntimeError("failed")
        with pytest.raises(RuntimeError):
            registry.load_pipeline("17")
    elif operation == "alias":
        client.set_registered_model_alias.side_effect = RuntimeError("failed")
        with pytest.raises(RuntimeError):
            registry.assign_alias("champion", "17")
    else:
        model = registry.log_model("lr", pipeline, {}, {}, lambda model: "pending card")
        mlflow.log_text.side_effect = RuntimeError("failed")
        with pytest.raises(RuntimeError):
            registry.record(model, "card", {})


def test_mlflow_import_is_optional(monkeypatch) -> None:
    import subprocess
    import sys

    command = [
        sys.executable,
        "-c",
        "import sys; import credit_scoring.training.main; import credit_scoring.registry.mlflow_adapter; assert 'mlflow' not in sys.modules",
    ]
    subprocess.run(command, check=True)  # noqa: S603
