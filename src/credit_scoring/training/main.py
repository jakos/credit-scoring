# ruff: noqa: T201
"""Entrypoint for running model training and reporting metrics."""

import argparse
from pathlib import Path

import pandas as pd

from credit_scoring.config import Settings
from credit_scoring.evaluation.metrics import evaluate_predictions
from credit_scoring.logging import configure_logging
from credit_scoring.training.tracking import log_run_to_mlflow
from credit_scoring.training.train import train


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model training."""
    parser = argparse.ArgumentParser(description="Train baseline credit scoring model")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
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
        default=None,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--lifecycle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train both families and run gated registry promotion (requires tracking URI).",
    )
    parser.add_argument("--registered-model-name", default=None)
    parser.add_argument("--promotion-policy-path", type=Path, default=None)
    args = parser.parse_args()
    overrides = {
        name: value
        for name, value in {
            "CREDIT_SCORING_MAX_ITER": args.max_iter,
            "CREDIT_SCORING_REGISTRY_ENABLED": args.lifecycle,
            "CREDIT_SCORING_REGISTERED_MODEL_NAME": args.registered_model_name,
            "CREDIT_SCORING_PROMOTION_POLICY_PATH": args.promotion_policy_path,
            "CREDIT_SCORING_MLFLOW_TRACKING_URI": args.mlflow_tracking_uri,
            "CREDIT_SCORING_MLFLOW_EXPERIMENT": args.mlflow_experiment,
        }.items()
        if value is not None
    }
    settings = Settings(**overrides)
    args.max_iter = settings.max_iter or 1000
    args.mlflow_tracking_uri = settings.mlflow_tracking_uri
    args.mlflow_experiment = settings.mlflow_experiment
    args.lifecycle = settings.registry_enabled
    if args.lifecycle and not settings.mlflow_tracking_uri:
        parser.error("Registry lifecycle requires a nonempty MLflow tracking URI")
    args.settings = settings
    return args


def main() -> None:
    """Run training pipeline and print evaluation metrics."""
    args = parse_args()

    settings = getattr(args, "settings", None) or Settings()
    logger = configure_logging(settings.log_level)
    if getattr(args, "lifecycle", False):
        from credit_scoring.data.loader import load_credit_data
        from credit_scoring.registry.lifecycle import run_lifecycle
        from credit_scoring.registry.mlflow_adapter import MLflowRegistry
        from credit_scoring.registry.policy import load_policy

        policy = load_policy(settings.promotion_policy_path)
        if not settings.mlflow_tracking_uri:
            raise ValueError("Registry lifecycle requires MLflow tracking")
        registry = MLflowRegistry(
            settings.mlflow_tracking_uri, settings.mlflow_experiment, settings.registered_model_name
        )
        decision = run_lifecycle(
            load_credit_data(settings.dataset)
            if settings.dataset is not None
            else load_credit_data(),
            registry,
            policy,
            max_iter=args.max_iter,
        )
        logger.info(
            "lifecycle_completed",
            extra={"fields": {"promoted": decision.approved, "reasons": decision.reasons}},
        )
        print(f"Lifecycle completed. Promoted: {decision.approved}")
        return
    if settings.dataset is None:
        artifacts = train(max_iter=args.max_iter)
    else:
        artifacts = train(max_iter=args.max_iter, dataset_path=settings.dataset)
    metrics = evaluate_predictions(
        y_true=pd.Series(artifacts.y_test),
        y_pred=pd.Series(artifacts.y_pred),
        y_proba=pd.Series(artifacts.y_proba),
    )

    logger.info(
        "training_completed",
        extra={"fields": {"sample_count": len(artifacts.y_test), "metrics": metrics}},
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
        logger.info("mlflow_logging_completed")
        print("MLflow logging completed.")


if __name__ == "__main__":
    main()
