"""Basic Airflow DAG for the credit-scoring project."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime


def _validate_dataset() -> None:
    """Ensure the source dataset exists before training starts."""
    dataset_path = Path(os.getenv("CREDIT_SCORING_DATASET", "data/raw/UCI_Credit_Card.csv"))
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set CREDIT_SCORING_DATASET to the correct path."
        )


def _run_training() -> None:
    """Run the existing training entrypoint from this repository."""
    max_iter = os.getenv("CREDIT_SCORING_MAX_ITER", "300")
    command = ["python", "-m", "credit_scoring.modeling.main", "--max-iter", str(max_iter)]

    # Use a plain Python module call so the same DAG works in local and containerized Airflow.
    subprocess.run(command, check=True)  # noqa: S603


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
