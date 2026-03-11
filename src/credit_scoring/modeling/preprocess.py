"""Data loading and preprocessing helpers for model training."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "default.payment.next.month"


def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split full DataFrame into features and target."""
    drop_columns = drop_columns or []
    columns_to_drop = [target_column, *drop_columns]
    x = df.drop(columns=columns_to_drop, errors="ignore")
    y = df[target_column]
    return x, y


def train_test_split_stratified(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split for classification."""
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """Build numeric preprocessing stage for sklearn pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
    )
