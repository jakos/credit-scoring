from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_scoring.config.settings import ServingSettings
from credit_scoring.features.preprocess import build_preprocessor
from credit_scoring.serving.contracts import ModelSnapshot
from credit_scoring.serving.loading import load_snapshot
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE


@pytest.fixture(scope="session")
def pipeline() -> Pipeline:
    rng = np.random.default_rng(42)
    names = list(APPLICANT_EXAMPLE["features"])
    frame = pd.DataFrame(rng.normal(size=(40, len(names))), columns=names)
    frame.loc[0, "AGE"] = np.nan
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(names)),
            ("model", LogisticRegression(random_state=42)),
        ]
    ).fit(frame, [0, 1] * 20)


@pytest.fixture(scope="session")
def model_directory(tmp_path_factory: pytest.TempPathFactory, pipeline: Pipeline) -> Path:
    path = tmp_path_factory.mktemp("artifact") / "model"
    mlflow.sklearn.save_model(pipeline, str(path), pip_requirements=[])
    return path


@pytest.fixture
def settings(model_directory: Path) -> ServingSettings:
    return ServingSettings(
        CREDIT_SCORING_SERVING_MODEL_SOURCE="local",
        CREDIT_SCORING_SERVING_MODEL_PATH=model_directory,
    )


@pytest.fixture
def snapshot(settings: ServingSettings) -> ModelSnapshot:
    return load_snapshot(settings)
