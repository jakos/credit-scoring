"""Airflow DAG for periodic retraining in the credit-scoring project."""

from __future__ import annotations

import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime

from credit_scoring.config import Settings
from credit_scoring.data.loader import DEFAULT_DATA_PATH
from credit_scoring.logging import configure_logging


def _validate_dataset() -> None:
    """Ensure retraining input dataset exists."""
    settings = Settings()
    logger = configure_logging(settings.log_level)
    dataset_path = settings.dataset if settings.dataset is not None else DEFAULT_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set CREDIT_SCORING_DATASET to the correct path."
        )

    logger.info("dataset_validated")


def _run_retraining() -> None:
    """Run baseline model retraining with optional MLflow logging."""
    settings = Settings()
    logger = configure_logging(settings.log_level)
    max_iter = settings.retrain_max_iter
    mlflow_tracking_uri = settings.mlflow_tracking_uri
    mlflow_experiment = settings.mlflow_experiment

    command = [
        "python",
        "-m",
        "credit_scoring.training.main",
        "--max-iter",
        str(max_iter),
    ]

    if mlflow_tracking_uri:
        command.extend(
            [
                "--enable-mlflow",
                "--mlflow-tracking-uri",
                mlflow_tracking_uri,
                "--mlflow-experiment",
                mlflow_experiment,
            ]
        )

    subprocess.run(command, check=True)  # noqa: S603
    logger.info("retraining_task_completed", extra={"fields": {"max_iter": max_iter}})


with DAG(
    dag_id="credit_scoring_retraining",
    description="Retrain the baseline credit-scoring model on a schedule.",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["credit-scoring", "retraining"],
) as dag:
    validate_dataset = PythonOperator(
        task_id="validate_dataset",
        python_callable=_validate_dataset,
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=_run_retraining,
    )

    validate_dataset >> retrain_model
