"""Optional MLflow integration."""

from importlib import import_module
from typing import Any


def log_run_to_mlflow(
    *,
    max_iter: int,
    metrics: dict[str, float],
    pipeline: Any,
    tracking_uri: str | None,
    experiment_name: str,
    run_name: str | None,
) -> None:
    """Log model training metadata and artifacts to MLflow."""
    try:
        mlflow = import_module("mlflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow is not installed. Install modeling dependencies with: uv sync --group modeling"
        ) from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("max_iter", max_iter)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
