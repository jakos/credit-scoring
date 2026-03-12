# ruff: noqa: T201
"""Entrypoint for running model training and reporting metrics."""

import argparse
from importlib import import_module
from typing import Any

import pandas as pd

from credit_scoring.modeling.evaluate import evaluate_predictions
from credit_scoring.modeling.train import train


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model training."""
    parser = argparse.ArgumentParser(description="Train baseline credit scoring model")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for LogisticRegression optimizer.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Enable MLflow tracking for params, metrics, and model artifact.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking URI (for example: http://127.0.0.1:5000).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="credit-scoring",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    return parser.parse_args()


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


def main() -> None:
    """Run training pipeline and print evaluation metrics."""
    args = parse_args()

    artifacts = train(max_iter=args.max_iter)
    metrics = evaluate_predictions(
        y_true=pd.Series(artifacts.y_test),
        y_pred=pd.Series(artifacts.y_pred),
        y_proba=pd.Series(artifacts.y_proba),
    )

    print("Model training completed.")
    print(f"Test samples: {len(artifacts.y_test)}")
    print("Evaluation metrics:")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"- {key}: {metrics[key]:.4f}")

    if args.enable_mlflow:
        log_run_to_mlflow(
            max_iter=args.max_iter,
            metrics=metrics,
            pipeline=artifacts.pipeline,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
        )
        print("MLflow logging completed.")


if __name__ == "__main__":
    main()
