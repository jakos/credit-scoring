from unittest.mock import MagicMock

import pytest

from credit_scoring.training import tracking


def test_logs_parameters_metrics_and_model(monkeypatch) -> None:
    mlflow = MagicMock()
    monkeypatch.setattr(tracking, "import_module", lambda _: mlflow)
    tracking.log_run_to_mlflow(
        max_iter=37,
        metrics={"accuracy": 0.8, "pr_auc": 0.9},
        pipeline="model",
        tracking_uri="http://localhost:5000",
        experiment_name="experiment",
        run_name="run",
    )
    mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
    mlflow.set_experiment.assert_called_once_with("experiment")
    mlflow.start_run.assert_called_once_with(run_name="run")
    mlflow.log_param.assert_called_once_with("max_iter", 37)
    assert mlflow.log_metric.call_args_list == [(("accuracy", 0.8),), (("pr_auc", 0.9),)]
    mlflow.sklearn.log_model.assert_called_once_with("model", artifact_path="model")


def test_missing_mlflow_has_install_hint(monkeypatch) -> None:
    def unavailable(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(tracking, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="uv sync --group modeling"):
        tracking.log_run_to_mlflow(
            max_iter=1,
            metrics={},
            pipeline=None,
            tracking_uri=None,
            experiment_name="experiment",
            run_name=None,
        )
