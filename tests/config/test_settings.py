import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_scoring.config import Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.dataset is None
    assert settings.max_iter is None
    assert settings.retrain_max_iter == 500
    assert settings.max_missing_ratio == 0.10
    assert settings.mlflow_tracking_uri is None
    assert settings.mlflow_experiment == "credit-scoring"


def test_all_environment_aliases(monkeypatch) -> None:
    values = {
        "APP_ENV": "test",
        "LOG_LEVEL": "debug",
        "CREDIT_SCORING_DATASET": "custom.csv",
        "CREDIT_SCORING_MAX_ITER": "23",
        "CREDIT_SCORING_RETRAIN_MAX_ITER": "24",
        "CREDIT_SCORING_MAX_MISSING_RATIO": "0.3",
        "CREDIT_SCORING_MLFLOW_TRACKING_URI": "http://localhost:5000",
        "CREDIT_SCORING_MLFLOW_EXPERIMENT": "experiment",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    settings = Settings()
    assert settings.app_env == "test"
    assert settings.log_level == "DEBUG"
    assert settings.dataset == Path("custom.csv")
    assert settings.max_iter == 23
    assert settings.retrain_max_iter == 24
    assert settings.max_missing_ratio == 0.3
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
    assert settings.mlflow_experiment == "experiment"


def test_unprefixed_environment_names_are_ignored(monkeypatch) -> None:
    values = {
        "DATASET": "unrelated.csv",
        "MAX_ITER": "7",
        "RETRAIN_MAX_ITER": "8",
        "MAX_MISSING_RATIO": "0.9",
        "MLFLOW_TRACKING_URI": "http://shared:5000",
        "MLFLOW_EXPERIMENT": "unrelated",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    settings = Settings()
    assert settings.dataset is None
    assert settings.max_iter is None
    assert settings.retrain_max_iter == 500
    assert settings.max_missing_ratio == 0.10
    assert settings.mlflow_tracking_uri is None
    assert settings.mlflow_experiment == "credit-scoring"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LOG_LEVEL", "trace"),
        ("CREDIT_SCORING_MAX_ITER", "0"),
        ("CREDIT_SCORING_MAX_ITER", "-1"),
        ("CREDIT_SCORING_RETRAIN_MAX_ITER", "0"),
        ("CREDIT_SCORING_RETRAIN_MAX_ITER", "-1"),
        ("CREDIT_SCORING_MAX_MISSING_RATIO", "-0.1"),
        ("CREDIT_SCORING_MAX_MISSING_RATIO", "1.1"),
    ],
)
def test_invalid_environment(key: str, value: str, monkeypatch) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("ratio", ["0", "1"])
def test_ratio_boundaries(ratio: str, monkeypatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_MAX_MISSING_RATIO", ratio)
    assert Settings().max_missing_ratio == float(ratio)


def test_import_does_not_validate_and_constructions_are_fresh(monkeypatch) -> None:
    import credit_scoring.config.settings as module

    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "invalid")
    importlib.reload(module)
    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "17")
    assert module.Settings().max_iter == 17
    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "18")
    assert module.Settings().max_iter == 18


def test_no_implicit_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("CREDIT_SCORING_MAX_ITER=9\nLOG_LEVEL=invalid\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().max_iter is None
    assert Settings().log_level == "INFO"


def test_registry_environment_and_override(monkeypatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_REGISTRY_ENABLED", "true")
    monkeypatch.setenv("CREDIT_SCORING_REGISTERED_MODEL_NAME", "risk")
    monkeypatch.setenv("CREDIT_SCORING_PROMOTION_POLICY_PATH", "custom.toml")
    settings = Settings()
    assert settings.registry_enabled
    assert settings.registered_model_name == "risk"
    assert settings.promotion_policy_path == Path("custom.toml")
    assert not Settings(CREDIT_SCORING_REGISTRY_ENABLED=False).registry_enabled
    with pytest.raises(ValidationError):
        Settings(CREDIT_SCORING_REGISTERED_MODEL_NAME=" ")
