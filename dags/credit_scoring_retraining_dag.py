"""Airflow DAG for periodic retraining in the credit-scoring project."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime


def _validate_dataset() -> None:
    """Ensure retraining input dataset exists."""
    dataset_path = Path(os.getenv("CREDIT_SCORING_DATASET", "data/raw/UCI_Credit_Card.csv"))
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set CREDIT_SCORING_DATASET to the correct path."
        )


def _run_retraining() -> None:
    """Run baseline model retraining with optional MLflow logging."""
    max_iter = os.getenv("CREDIT_SCORING_RETRAIN_MAX_ITER", "500")
    mlflow_tracking_uri = os.getenv("CREDIT_SCORING_MLFLOW_TRACKING_URI")
    mlflow_experiment = os.getenv("CREDIT_SCORING_MLFLOW_EXPERIMENT", "credit-scoring")

    command = [
        "python",
        "-m",
        "credit_scoring.modeling.main",
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

    subprocess.run(command, check=True)


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
