"""Basic Airflow DAG for the credit-scoring project."""

from __future__ import annotations

import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime

from credit_scoring.config import Settings
from credit_scoring.data.loader import DEFAULT_DATA_PATH
from credit_scoring.logging import configure_logging


def _validate_dataset() -> None:
    """Ensure the source dataset exists before training starts."""
    settings = Settings()
    logger = configure_logging(settings.log_level)
    dataset_path = settings.dataset if settings.dataset is not None else DEFAULT_DATA_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set CREDIT_SCORING_DATASET to the correct path."
        )

    logger.info("dataset_validated")


def _run_training() -> None:
    """Run the existing training entrypoint from this repository."""
    settings = Settings()
    logger = configure_logging(settings.log_level)
    max_iter = settings.max_iter or 300
    command = ["python", "-m", "credit_scoring.training.main", "--max-iter", str(max_iter)]

    if settings.registry_enabled:
        command.append("--lifecycle")

    # Use a plain Python module call so the same DAG works in local and containerized Airflow.
    subprocess.run(command, check=True)  # noqa: S603
    logger.info("training_task_completed", extra={"fields": {"max_iter": max_iter}})


with DAG(
    dag_id="credit_scoring_training",
    description="DAG that validates input data and runs model training.",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["credit-scoring", "training"],
) as dag:
    validate_dataset = PythonOperator(
        task_id="validate_dataset",
        python_callable=_validate_dataset,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_run_training,
    )

    validate_dataset >> train_model
