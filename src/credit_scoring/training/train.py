"""Training helpers for credit default model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_scoring.data.loader import load_credit_data
from credit_scoring.features.preprocess import (
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
    *,
    dataset_path: Path | str | None = None,
) -> TrainingArtifacts:
    """Train baseline model and return artifacts for evaluation/reporting."""
    if df is None:
        df = load_credit_data(dataset_path) if dataset_path is not None else load_credit_data()

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


@dataclass
class LifecycleArtifacts:
    pipeline: Pipeline
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.Series
    train_scores: np.ndarray
    test_scores: np.ndarray
    predictions: np.ndarray


def train_lifecycle(df: pd.DataFrame, max_iter: int = 1000) -> dict[str, LifecycleArtifacts]:
    """Fit both families on one split, with independent train-only preprocessing."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = split_features_target(df, drop_columns=["ID"])
    x_train, x_test, y_train, y_test = train_test_split_stratified(x, y)
    numeric = x_train.select_dtypes(include="number").columns.tolist()
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=max_iter, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=50, max_leaf_nodes=15, early_stopping=False, random_state=42
        ),
    }
    results = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline([("preprocessor", build_preprocessor(numeric)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        results[name] = LifecycleArtifacts(
            pipeline,
            x_train,
            x_test,
            y_test,
            pipeline.predict_proba(x_train)[:, 1],
            pipeline.predict_proba(x_test)[:, 1],
            np.asarray(pipeline.predict(x_test)),
        )
    return results
