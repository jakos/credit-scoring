from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_scoring.config.settings import ServingSettings


def test_sources_and_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="tracking URI"):
        ServingSettings()
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", "http://registry")
    settings = ServingSettings()
    assert settings.serving_model_source == "registry"
    assert (settings.serving_host, settings.serving_port) == ("127.0.0.1", 8000)
    assert (settings.risk_medium_threshold, settings.risk_high_threshold) == (0.2, 0.5)
    assert settings.registered_model_name == "credit-scoring"
    monkeypatch.setenv("CREDIT_SCORING_SERVING_MODEL_SOURCE", "local")
    with pytest.raises(ValidationError, match="directory"):
        ServingSettings()
    monkeypatch.setenv("CREDIT_SCORING_SERVING_MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("CREDIT_SCORING_SERVING_HOST", "localhost")
    monkeypatch.setenv("CREDIT_SCORING_SERVING_PORT", "9000")
    monkeypatch.setenv("CREDIT_SCORING_RISK_MEDIUM_THRESHOLD", "0.1")
    monkeypatch.setenv("CREDIT_SCORING_RISK_HIGH_THRESHOLD", "0.8")
    settings = ServingSettings(CREDIT_SCORING_SERVING_PORT=9001)
    assert settings.serving_model_path == tmp_path
    assert settings.serving_host == "localhost"
    assert settings.serving_port == 9001
    assert (settings.risk_medium_threshold, settings.risk_high_threshold) == (0.1, 0.8)


@pytest.mark.parametrize(
    "overrides",
    [
        {"CREDIT_SCORING_SERVING_MODEL_SOURCE": "other"},
        {"CREDIT_SCORING_SERVING_MODEL_SOURCE": "local"},
        {"CREDIT_SCORING_MLFLOW_TRACKING_URI": " "},
        {"CREDIT_SCORING_SERVING_PORT": 0},
        {"CREDIT_SCORING_SERVING_PORT": 65536},
        {"CREDIT_SCORING_RISK_MEDIUM_THRESHOLD": 0},
        {"CREDIT_SCORING_RISK_HIGH_THRESHOLD": 1},
        {"CREDIT_SCORING_RISK_MEDIUM_THRESHOLD": 0.5},
        {"CREDIT_SCORING_RISK_MEDIUM_THRESHOLD": 0.6},
        {"CREDIT_SCORING_RISK_HIGH_THRESHOLD": float("nan")},
        {"CREDIT_SCORING_RISK_MEDIUM_THRESHOLD": float("inf")},
    ],
)
def test_invalid_serving_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ServingSettings(**{"CREDIT_SCORING_MLFLOW_TRACKING_URI": "http://registry", **overrides})
