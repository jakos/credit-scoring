# ruff: noqa: T201
"""Entrypoint for running model training and reporting metrics."""

import argparse

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
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
