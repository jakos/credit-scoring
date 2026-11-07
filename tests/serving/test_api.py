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
        explanation = single.json()["explanation"]
        assert set(explanation) == {
            "method",
            "baseline_log_odds",
            "top_features",
            "remaining_contribution",
        }
        assert explanation["method"] == "linear_log_odds"
        assert all(
            set(feature) == {"feature", "contribution"} for feature in explanation["top_features"]
        )
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
        assert "explanation" in models["ScoreResponse"]["required"]
        assert models["ScoreResponse"]["properties"]["explanation"]["$ref"].endswith("/Explanation")
        assert set(models["Explanation"]["required"]) == {
            "method",
            "baseline_log_odds",
            "top_features",
            "remaining_contribution",
        }
        assert set(models["FeatureContribution"]["required"]) == {"feature", "contribution"}
        assert models["BatchResponse"]["properties"]["scores"]["items"]["$ref"].endswith(
            "/ScoreResponse"
        )
        numeric = models["Applicant"]["properties"]["features"]["additionalProperties"]["anyOf"]
        assert all(
            branch["minimum"] == -1e12 and branch["maximum"] == 1e12
            for branch in numeric
            if branch["type"] in {"number", "integer"}
        )
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
    details = response.json()["detail"]
    assert all(d["loc"][: len(prefix) + 2] == [*prefix, "features", "AGE"] for d in details)
    assert any(d["type"] == "less_than_equal" for d in details)
    predictor.predict_proba.assert_not_called()
    assert "serving_prediction_failed" not in capsys.readouterr().err


def test_prediction_failure_redacted(
    settings: ServingSettings, snapshot: ModelSnapshot, capsys: pytest.CaptureFixture[str]
) -> None:
    predictor = Mock()
    predictor.predict_proba.side_effect = RuntimeError(
        f"secret-applicant-and-path {APPLICANT_EXAMPLE['features']}"
    )
    loader = Mock(return_value=ModelSnapshot(predictor, snapshot.features, 1))
    with TestClient(create_app(settings, loader)) as client:
        response = client.post("/v1/score", json=APPLICANT_EXAMPLE)
        assert response.status_code == 500
        assert response.json() == {"detail": "Model prediction failed"}
    log = capsys.readouterr().err
    assert "serving_prediction_failed" in log
    assert "secret-applicant-and-path" not in log
    assert "LIMIT_BAL" not in log
    assert '"cause_type": "RuntimeError"' in log
    assert '"cause_message": "Failure details redacted"' in log
    assert "Traceback" not in log


def test_invalid_settings_fail_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_SERVING_PORT", "0")
    with pytest.raises(ValidationError):
        create_app()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("value", [1e308, -1e308, 10**400, -(10**400)])
def test_extreme_finite_inputs_never_reach_prediction(
    settings: ServingSettings, snapshot: ModelSnapshot, batch: bool, value: float | int
) -> None:
    predictor = Mock()
    loader = Mock(return_value=ModelSnapshot(predictor, snapshot.features, 1))
    applicant = {"features": {**APPLICANT_EXAMPLE["features"], "AGE": value}}
    payload = {"applicants": [APPLICANT_EXAMPLE, applicant]} if batch else applicant
    with TestClient(create_app(settings, loader)) as client:
        response = client.post("/v1/score/batch" if batch else "/v1/score", json=payload)
    assert response.status_code == 422
    predictor.predict_proba.assert_not_called()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("value", [-1e12, 1e12])
def test_http_accepts_numeric_boundaries(
    settings: ServingSettings, batch: bool, value: float
) -> None:
    applicant = {"features": {**APPLICANT_EXAMPLE["features"], "AGE": value}}
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/score/batch" if batch else "/v1/score",
            json={"applicants": [applicant]} if batch else applicant,
        )
    assert response.status_code == 200


@pytest.mark.parametrize("batch", [False, True])
def test_raw_integer_beyond_python_parser_limit_returns_400(
    settings: ServingSettings, batch: bool
) -> None:
    import sys

    assert sys.get_int_max_str_digits() == 4300
    applicant = b'{"features":{"AGE":' + b"9" * 4301 + b"}}"
    raw = b'{"applicants":[' + applicant + b"]}" if batch else applicant
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/score/batch" if batch else "/v1/score",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400
    assert "999999" not in response.text


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("invalid_values", [False, True])
def test_unknown_feature_locations_are_distinct_and_private(
    settings: ServingSettings, batch: bool, invalid_values: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    features = {
        "unknown-z-sentinel": {"value-sentinel": "private"} if invalid_values else 1,
        **APPLICANT_EXAMPLE["features"],
        "unknown-a-sentinel": "value-sentinel" if invalid_values else 2,
    }
    applicant = {"features": features}
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/score/batch" if batch else "/v1/score",
            json={"applicants": [APPLICANT_EXAMPLE, applicant]} if batch else applicant,
        )
    assert response.status_code == 422
    prefix = ["body", "applicants", 1] if batch else ["body"]
    assert len(response.json()["detail"]) == 2
    assert {tuple(d["loc"]) for d in response.json()["detail"]} == {
        tuple([*prefix, "features", 0]),
        tuple([*prefix, "features", len(features) - 1]),
    }
    assert "sentinel" not in response.text + capsys.readouterr().err


@pytest.mark.parametrize(
    "failure,kind,message",
    [
        ("missing", "ModelContractError", "No champion version is available"),
        ("incompatible", "ModelContractError", "Required input feature metadata is missing"),
        ("connection", "RegistryLoadError", "Registry lookup or model loading failed"),
    ],
)
def test_startup_diagnostics_distinguish_failures(
    settings: ServingSettings,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    kind: str,
    message: str,
) -> None:
    import json

    from credit_scoring.serving.loading import load_snapshot

    def loader(configuration: ServingSettings) -> ModelSnapshot:
        if failure == "incompatible":
            return validate_pipeline(object())
        registry = Mock()
        registry.resolve_alias.return_value = None
        if failure == "connection":
            registry.resolve_alias.side_effect = ConnectionError(
                "https://user:credential-sentinel@tracking-sentinel/unknown-key-sentinel"
            )
        return load_snapshot(
            ServingSettings(CREDIT_SCORING_MLFLOW_TRACKING_URI="http://tracking-sentinel"),
            registry=registry,
        )

    with TestClient(create_app(settings, loader)) as client:
        assert client.get("/readyz").status_code == 503
    serialized = capsys.readouterr().err
    assert "sentinel" not in serialized
    record = next(
        json.loads(line)
        for line in serialized.splitlines()
        if json.loads(line)["event"] == "serving_model_load_failed"
    )
    assert record["exception_type"] == kind
    assert record["message"] == message
    if failure == "connection":
        assert record["cause_type"] == "ConnectionError"
