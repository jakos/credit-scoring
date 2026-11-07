from dataclasses import replace
from unittest.mock import Mock

from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.parser import text_string_to_metric_families
from pydantic import SecretStr

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.app import create_app
from credit_scoring.serving.contracts import ModelSnapshot
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE


def test_metrics_status_deltas_and_isolation(
    settings: ServingSettings, snapshot: ModelSnapshot
) -> None:
    settings = settings.model_copy(update={"serving_api_key": SecretStr("synthetic-key")})
    first = create_app(settings, lambda _: snapshot)
    second = create_app(settings, lambda _: snapshot)
    registry = first.state.metrics.registry
    with TestClient(first) as client, TestClient(second) as other:
        auth = {"X-API-Key": "synthetic-key"}
        calls = [
            ("GET", "/healthz", None, 200),
            ("POST", "/v1/score", APPLICANT_EXAMPLE, 401),
            ("POST", "/v1/score", {}, 422),
            ("GET", "/absent-sensitive-query?sentinel=private", None, 404),
        ]
        for method, path, body, status in calls:
            headers = {} if status == 401 else auth
            response = client.request(method, path, json=body, headers=headers)
            assert response.status_code == status
            route = "/healthz" if status == 200 else "unmatched" if status == 404 else "/v1/score"
            assert (
                registry.get_sample_value(
                    "credit_scoring_http_requests_total",
                    {"method": method, "route": route, "status": str(status)},
                )
                == 1
            )
        assert (
            client.post(
                "/v1/score/batch", json={"applicants": [APPLICANT_EXAMPLE] * 3}, headers=auth
            ).status_code
            == 200
        )
        assert registry.get_sample_value("credit_scoring_probability_of_default_count") == 3
        first.state.scorer = None
        assert client.post("/v1/score", json=APPLICANT_EXAMPLE, headers=auth).status_code == 503
        bad = Mock()
        bad.predict_proba.side_effect = RuntimeError("sensitive")
        from credit_scoring.serving.scoring import Scorer

        first.state.scorer = Scorer(replace(snapshot, predictor=bad), 0.2, 0.5)
        assert client.post("/v1/score", json=APPLICANT_EXAMPLE, headers=auth).status_code == 500
        for status in (401, 422, 503, 500):
            assert (
                registry.get_sample_value(
                    "credit_scoring_http_requests_total",
                    {"method": "POST", "route": "/v1/score", "status": str(status)},
                )
                == 1
            )
        assert registry.get_sample_value("credit_scoring_probability_of_default_count") == 3
        assert (
            registry.get_sample_value(
                "credit_scoring_http_request_duration_seconds_count",
                {"method": "POST", "route": "/v1/score"},
            )
            == 4
        )
        assert (
            registry.get_sample_value(
                "credit_scoring_http_request_duration_seconds_count",
                {"method": "POST", "route": "/v1/score/batch"},
            )
            == 1
        )
        before = list(registry.collect())
        scrape = client.get("/metrics")
        assert scrape.headers["content-type"] == CONTENT_TYPE_LATEST
        assert list(text_string_to_metric_families(scrape.text))
        client.get("/metrics")
        assert list(registry.collect()) == before
        assert "sensitive" not in scrape.text and "private" not in scrape.text
        assert "synthetic-key" not in scrape.text
        assert (
            second.state.metrics.registry.get_sample_value(
                "credit_scoring_probability_of_default_count"
            )
            == 0
        )
        assert "credit_scoring_http_requests_total{" not in other.get("/metrics").text
        for name, labels, buckets in [
            (
                "credit_scoring_http_request_duration_seconds",
                {"method": "POST", "route": "/v1/score"},
                [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            ),
            (
                "credit_scoring_probability_of_default",
                {},
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            ),
        ]:
            samples = [
                s
                for family in registry.collect()
                for s in family.samples
                if s.name == name + "_bucket" and all(s.labels[k] == v for k, v in labels.items())
            ]
            assert [float(s.labels["le"]) for s in samples] == [*buckets, float("inf")]
            assert registry.get_sample_value(name + "_sum", labels) >= 0


def test_unknown_methods_and_paths_have_bounded_labels(settings: ServingSettings) -> None:
    app = create_app(settings)
    with TestClient(app) as client:
        for index in range(4):
            assert (
                client.request("PRIVATE" + str(index), "/private/" + str(index)).status_code == 404
            )
        text = client.get("/metrics").text
        assert 'method="OTHER",route="unmatched",status="404"} 4.0' in text
        assert "PRIVATE" not in text and "/private/" not in text


def test_response_failure_preserves_sent_status_and_excludes_scores() -> None:
    import asyncio

    import pytest
    from fastapi import FastAPI
    from starlette.types import Message, Receive, Scope, Send

    from credit_scoring.serving.metrics import Metrics
    from credit_scoring.serving.middleware import RequestMiddleware

    async def fail(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        raise RuntimeError("private-response-sentinel")

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    metrics = Metrics()
    middleware = RequestMiddleware(fail, metrics=metrics, logger=Mock(), key=None)
    scope: Scope = {
        "type": "http",
        "app": FastAPI(),
        "path": "/",
        "method": "POST",
        "headers": [],
        "state": {"score_probabilities": [0.5]},
    }
    with pytest.raises(RuntimeError, match="private-response-sentinel"):
        asyncio.run(middleware(scope, Mock(), send))
    assert len(messages) == 2
    assert (
        metrics.registry.get_sample_value(
            "credit_scoring_http_requests_total",
            {"method": "POST", "route": "unmatched", "status": "202"},
        )
        == 1
    )
    assert metrics.registry.get_sample_value("credit_scoring_probability_of_default_count") == 0
