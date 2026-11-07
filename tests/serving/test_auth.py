from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.app import create_app
from credit_scoring.serving.auth import authorized
from credit_scoring.serving.contracts import ModelSnapshot
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE


@pytest.mark.parametrize("enabled", [False, True])
def test_authentication_and_openapi(
    settings: ServingSettings, snapshot: ModelSnapshot, enabled: bool
) -> None:
    key = "synthetic-auth-sentinel"
    settings = settings.model_copy(update={"serving_api_key": SecretStr(key) if enabled else None})
    predictor = Mock(wraps=snapshot.predictor)
    app = create_app(settings, lambda _: replace(snapshot, predictor=predictor))
    with TestClient(app) as client:
        for path in ("/healthz", "/readyz", "/metrics"):
            assert client.get(path).status_code == 200
        for headers in ({}, {"X-API-Key": "incorrect"}, [("X-API-Key", key), ("X-API-Key", key)]):
            response = client.post("/v1/score", json=APPLICANT_EXAMPLE, headers=headers)
            assert response.status_code == (401 if enabled else 200)
            if enabled:
                assert response.json() == {"detail": "Unauthorized"}
                assert "X-Request-ID" in response.headers
        if enabled:
            predictor.predict_proba.assert_not_called()
        assert (
            client.post("/v1/score", json=APPLICANT_EXAMPLE, headers={"X-API-Key": key}).status_code
            == 200
        )
        for path in (
            "/docs",
            "/redoc",
            "/openapi.json",
            "/docs/oauth2-redirect",
            "/absent",
            "/healthz/",
        ):
            assert client.get(path, follow_redirects=False).status_code == (
                401
                if enabled
                else 404
                if path == "/absent"
                else 307
                if path == "/healthz/"
                else 200
            )
        schema = client.get("/openapi.json", headers={"X-API-Key": key}).json()
        for path in ("/v1/score", "/v1/score/batch"):
            assert schema["paths"][path]["post"].get("security", []) == (
                [{"APIKey": []}] if enabled else []
            )
        if enabled:
            assert schema["components"]["securitySchemes"]["APIKey"] == {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            }


def test_unicode_key_comparison_uses_utf8() -> None:
    key = "synthetic-za\u017c\u00f3\u0142\u0107".encode()
    scope = {"path": "/v1/score", "headers": [(b"x-api-key", key)]}
    assert authorized(scope, key)
    assert not authorized(scope, b"incorrect")
