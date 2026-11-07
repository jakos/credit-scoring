"""Keep application configuration tests independent of the developer environment."""

import os

import pytest


@pytest.fixture(autouse=True)
def clean_application_environment(monkeypatch) -> None:
    for key in os.environ:
        if key.startswith("CREDIT_SCORING_") or key in {"APP_ENV", "LOG_LEVEL"}:
            monkeypatch.delenv(key)
