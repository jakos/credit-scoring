"""Training helpers for credit default model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data_loader import load_credit_data
from .preprocess import (
    build_preprocessor,
    split_features_target,
    train_test_split_stratified,
)


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    x_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray


def train(
    df: pd.DataFrame | None = None,
    max_iter: int = 1000,
) -> TrainingArtifacts:
    """Train baseline model and return artifacts for evaluation/reporting."""
    if df is None:
        df = load_credit_data()

    x, y = split_features_target(df, drop_columns=["ID"])

    x_train, x_test, y_train, y_test = train_test_split_stratified(x, y)

    # Keep only numeric columns for current preprocessor implementation
    numeric_columns = x_train.select_dtypes(include="number").columns.tolist()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_columns)),
            ("model", LogisticRegression(max_iter=max_iter, random_state=42)),
        ]
    )

    pipeline.fit(x_train, y_train)

    y_pred = np.asarray(pipeline.predict(x_test))
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    return TrainingArtifacts(
        pipeline=pipeline,
        x_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )
