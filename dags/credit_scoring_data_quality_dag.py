"""Airflow DAG for dataset quality checks in the credit-scoring project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from pendulum import datetime

from credit_scoring.config import Settings
from credit_scoring.data.loader import DEFAULT_DATA_PATH
from credit_scoring.logging import configure_logging

REQUIRED_COLUMNS = {
    "ID",
    "default.payment.next.month",
    "LIMIT_BAL",
    "AGE",
}


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set CREDIT_SCORING_DATASET to the correct path."
        )
    return pd.read_csv(dataset_path)


def _validate_schema() -> None:
    """Check required columns used by baseline training."""
    settings = Settings()
    logger = configure_logging(settings.log_level)
    df = _load_dataset(settings.dataset if settings.dataset is not None else DEFAULT_DATA_PATH)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    logger.info("schema_validated", extra={"fields": {"sample_count": len(df)}})


def _validate_missing_ratio() -> None:
    """Fail if overall missing ratio exceeds configured threshold."""
    settings = Settings()
    logger = configure_logging(settings.log_level)
    df = _load_dataset(settings.dataset if settings.dataset is not None else DEFAULT_DATA_PATH)
    threshold = settings.max_missing_ratio
    missing_ratio = float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]))

    if missing_ratio > threshold:
        raise ValueError(
            f"Missing ratio {missing_ratio:.4f} is above allowed threshold {threshold:.4f}."
        )

    logger.info(
        "missing_ratio_validated",
        extra={"fields": {"missing_ratio": missing_ratio, "threshold": threshold}},
    )


with DAG(
    dag_id="credit_scoring_data_quality",
    description="Validate dataset schema and missing-data ratio before model workflows.",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["credit-scoring", "data-quality"],
) as dag:
    validate_schema = PythonOperator(
        task_id="validate_schema",
        python_callable=_validate_schema,
    )

    validate_missing_ratio = PythonOperator(
        task_id="validate_missing_ratio",
        python_callable=_validate_missing_ratio,
    )

    validate_schema >> validate_missing_ratio
