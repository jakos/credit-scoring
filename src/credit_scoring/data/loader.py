"""Data loading helpers for model training."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"


def load_credit_data(csv_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the default credit card dataset from CSV."""
    return pd.read_csv(csv_path)
