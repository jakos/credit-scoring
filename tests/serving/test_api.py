from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.app import create_app
from credit_scoring.serving.contracts import ModelSnapshot
from credit_scoring.serving.loading import validate_pipeline
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE


def test_single_batch_and_openapi(settings: ServingSettings, snapshot: ModelSnapshot) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").json() == {"status": "live"}
        assert client.get("/readyz").json() == {"status": "ready"}
        single = client.post("/v1/score", json=APPLICANT_EXAMPLE)
        assert single.status_code == 200
        frame = pd.DataFrame([APPLICANT_EXAMPLE["features"]], columns=snapshot.features.names)
        expected = snapshot.predictor.predict_proba(frame)[0, snapshot.default_column]
        assert single.json()["probability_of_default"] == expected
        batch = client.post("/v1/score/batch", json={"applicants": [APPLICANT_EXAMPLE] * 2})
        assert batch.status_code == 200
        assert batch.json() == {"scores": [single.json()] * 2}
        schema = client.get("/openapi.json").json()
        assert set(schema["paths"]) == {"/healthz", "/readyz", "/v1/score", "/v1/score/batch"}
        for path in ("/v1/score", "/v1/score/batch"):
            operation = schema["paths"][path]["post"]
            assert set(operation["responses"]) == {"200", "422", "500", "503"}
            assert "requestBody" in operation
        assert "503" in schema["paths"]["/readyz"]["get"]["responses"]
        models = schema["components"]["schemas"]
        assert {
            "Applicant",
            "BatchRequest",
            "BatchResponse",
            "ScoreResponse",
            "ErrorResponse",
            "HealthResponse",
        } <= models.keys()
        example = models["Applicant"]["examples"][0]
        assert example == APPLICANT_EXAMPLE
        assert client.post("/v1/score", json=example).status_code == 200
        assert client.get("/docs").status_code == 200


def test_lifespan_snapshot_is_fixed_and_cleared(
    settings: ServingSettings, snapshot: ModelSnapshot
) -> None:
    loader = Mock(return_value=snapshot)
    app = create_app(settings, loader)
    with TestClient(app) as before_startup:
        assert before_startup.get("/readyz").status_code == 200
        for _ in range(2):
            assert before_startup.post("/v1/score", json=APPLICANT_EXAMPLE).status_code == 200
        loader.assert_called_once_with(settings)
    assert app.state.scorer is None
    with TestClient(app):
        assert loader.call_count == 2
    assert app.state.scorer is None


def test_before_startup_unready(settings: ServingSettings) -> None:
    loader = Mock()
    client = TestClient(create_app(settings, loader))
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503
    assert client.post("/v1/score", json=APPLICANT_EXAMPLE).status_code == 503
    loader.assert_not_called()


def test_load_failure_live_unready_and_redacted(
    settings: ServingSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "https://user:password@private/secret/path"  # noqa: S105 -- synthetic redaction probe
    loader = Mock(side_effect=RuntimeError(secret))
    with TestClient(create_app(settings, loader)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        for path, body in [
            ("/v1/score", APPLICANT_EXAMPLE),
            ("/v1/score/batch", {"applicants": [APPLICANT_EXAMPLE]}),
        ]:
            response = client.post(path, json=body)
            assert response.status_code == 503
            assert response.json() == {"detail": "Model is not ready"}
        loader.assert_called_once()
    log = capsys.readouterr().err
    assert "serving_model_load_failed" in log
    assert secret not in log
    assert "Traceback" not in log


def test_incompatible_model_prevents_readiness(settings: ServingSettings) -> None:
    def incompatible(configuration: ServingSettings) -> ModelSnapshot:
        return validate_pipeline(object())

    with TestClient(create_app(settings, incompatible)) as client:
        assert client.get("/readyz").status_code == 503


@pytest.mark.parametrize(
    "features,reason,location",
    [
        ({}, "missing", ["features", "LIMIT_BAL"]),
        ({**APPLICANT_EXAMPLE["features"], "secret": 123456789}, "extra_forbidden", ["features"]),
        (
            {**APPLICANT_EXAMPLE["features"], "AGE": "private-record"},
            "float_type",
            ["features", "AGE"],
        ),
        ({**APPLICANT_EXAMPLE["features"], "AGE": True}, "float_type", ["features", "AGE"]),
    ],
)
def test_invalid_single_and_batch(
    settings: ServingSettings, features: dict[str, object], reason: str, location: list[str]
) -> None:
    with TestClient(create_app(settings)) as client:
        for path, payload, prefix in [
            ("/v1/score", {"features": features}, ["body"]),
            (
                "/v1/score/batch",
                {"applicants": [APPLICANT_EXAMPLE, {"features": features}]},
                ["body", "applicants", 1],
            ),
        ]:
            response = client.post(path, json=payload)
            assert response.status_code == 422
            details = response.json()["detail"]
            assert any(
                d["loc"][: len(prefix + location)] == prefix + location and d["type"] == reason
                for d in details
            )
            assert "scores" not in response.json()
            assert "private-record" not in response.text
            assert "123456789" not in response.text
            assert all("input" not in d and "ctx" not in d for d in details)


def test_null_supported(settings: ServingSettings) -> None:
    features = {**APPLICANT_EXAMPLE["features"], "AGE": None}
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/score", json={"features": features}).status_code == 200


@pytest.mark.parametrize("batch", [False, True])
def test_oversized_integer_is_validation_error(
    settings: ServingSettings,
    snapshot: ModelSnapshot,
    capsys: pytest.CaptureFixture[str],
    batch: bool,
) -> None:
    predictor = Mock()
    loader = Mock(return_value=ModelSnapshot(predictor, snapshot.features, 1))
    applicant = {"features": {**APPLICANT_EXAMPLE["features"], "AGE": 10**400}}
    path = "/v1/score/batch" if batch else "/v1/score"
    payload = {"applicants": [APPLICANT_EXAMPLE, applicant]} if batch else applicant
    prefix: list[str | int] = ["body", "applicants", 1] if batch else ["body"]
    with TestClient(create_app(settings, loader)) as client:
        response = client.post(path, json=payload)
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": [*prefix, "features", "AGE"],
                "msg": "Model feature must be representable as a finite float",
                "type": "finite_number",
            }
        ]
    }
    predictor.predict_proba.assert_not_called()
    assert "serving_prediction_failed" not in capsys.readouterr().err


def test_prediction_failure_redacted(
    settings: ServingSettings, snapshot: ModelSnapshot, capsys: pytest.CaptureFixture[str]
) -> None:
    predictor = Mock()
    predictor.predict_proba.side_effect = RuntimeError("secret-applicant-and-path")
    loader = Mock(return_value=ModelSnapshot(predictor, snapshot.features, 1))
    with TestClient(create_app(settings, loader)) as client:
        response = client.post("/v1/score", json=APPLICANT_EXAMPLE)
        assert response.status_code == 500
        assert response.json() == {"detail": "Model prediction failed"}
    log = capsys.readouterr().err
    assert "serving_prediction_failed" in log
    assert "secret-applicant-and-path" not in log
    assert "LIMIT_BAL" not in log
    assert "Traceback" not in log


def test_invalid_settings_fail_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_SERVING_PORT", "0")
    with pytest.raises(ValidationError):
        create_app()
