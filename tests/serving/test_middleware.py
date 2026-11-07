import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.app import create_app
from credit_scoring.serving.contracts import ModelSnapshot
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE


@pytest.mark.parametrize("incoming", [None, "", "bad id?", "x" * 129, "a.b_C-9", "x" * 128])
def test_request_id_validation(
    settings: ServingSettings, incoming: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.get(
            "/healthz", headers={} if incoming is None else {"X-Request-ID": incoming}
        )
    resolved = response.headers["X-Request-ID"]
    if incoming in ("a.b_C-9", "x" * 128):
        assert resolved == incoming
    else:
        UUID(resolved)
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    access = [r for r in records if r["event"] == "serving_access"]
    assert len(access) == 1 and access[0]["request_id"] == resolved


def test_combined_statuses_logs_and_redaction(
    settings: ServingSettings, snapshot: ModelSnapshot, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = settings.model_copy(update={"serving_api_key": SecretStr("credential-sentinel")})
    app = create_app(settings, lambda _: snapshot)
    with TestClient(app) as client:
        for index, (path, headers, body, status) in enumerate(
            [
                ("/v1/score", {}, APPLICANT_EXAMPLE, 401),
                (
                    "/v1/score",
                    {"X-API-Key": "credential-sentinel"},
                    {"features": {"AGE": "applicant-sentinel"}},
                    422,
                ),
                ("/metrics", {}, None, 200),
                ("/v1/score", {"X-API-Key": "credential-sentinel"}, APPLICANT_EXAMPLE, 200),
            ]
        ):
            response = client.request(
                "GET" if body is None else "POST",
                path + "?query-sentinel=1",
                headers={**headers, "X-Request-ID": f"request-{index}"},
                json=body,
            )
            assert response.status_code == status
            assert response.headers["X-Request-ID"] == f"request-{index}"
        app.state.scorer = None
        response = client.post(
            "/v1/score",
            json=APPLICANT_EXAMPLE,
            headers={"X-API-Key": "credential-sentinel", "X-Request-ID": "unready"},
        )
        assert response.status_code == 503 and response.headers["X-Request-ID"] == "unready"
    log = capsys.readouterr().err
    for sentinel in ("credential-sentinel", "applicant-sentinel", "query-sentinel"):
        assert sentinel not in log
    access = [
        r for line in log.splitlines() if (r := json.loads(line))["event"] == "serving_access"
    ]
    assert len(access) == 5
    assert [r["status"] for r in access] == [401, 422, 200, 200, 503]
    assert all(
        set(r)
        == {
            "timestamp",
            "severity",
            "logger",
            "event",
            "request_id",
            "method",
            "route",
            "status",
            "duration_seconds",
        }
        for r in access
    )
    assert all(r["duration_seconds"] >= 0 for r in access)


def test_concurrent_request_ids(
    settings: ServingSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(create_app(settings)) as client:

        def request(index: int) -> str:
            return client.get("/healthz", headers={"X-Request-ID": f"parallel-{index}"}).headers[
                "X-Request-ID"
            ]

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(request, range(24))) == [f"parallel-{i}" for i in range(24)]
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert {r["request_id"] for r in records if r["event"] == "serving_access"} == {
        f"parallel-{i}" for i in range(24)
    }


def test_unexpected_error_is_sanitized_and_observed(
    settings: ServingSettings, capsys: pytest.CaptureFixture[str]
) -> None:
    app = create_app(settings)

    @app.get("/failure")
    def fail() -> None:
        raise RuntimeError("exception-sentinel")

    with TestClient(app) as client:
        response = client.get("/failure", headers={"X-Request-ID": "failure-id"})
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error"}
        assert response.headers["X-Request-ID"] == "failure-id"
        assert (
            app.state.metrics.registry.get_sample_value(
                "credit_scoring_http_requests_total",
                {"method": "GET", "route": "/failure", "status": "500"},
            )
            == 1
        )
    log = capsys.readouterr().err
    assert "exception-sentinel" not in log
    records = [json.loads(line) for line in log.splitlines()]
    assert len([r for r in records if r["event"] == "serving_access"]) == 1


def test_monotonic_duration_and_non_ascii_id() -> None:
    import asyncio

    from fastapi import FastAPI

    from credit_scoring.serving.metrics import Metrics
    from credit_scoring.serving.middleware import RequestMiddleware

    async def endpoint(scope: dict, receive: object, send: object) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = []

    async def send(message: dict) -> None:
        messages.append(message)

    metrics = Metrics()
    middleware = RequestMiddleware(
        endpoint, metrics=metrics, logger=Mock(), key=None, clock=Mock(side_effect=[10.0, 10.25])
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "app": FastAPI(),
                "path": "/",
                "method": "GET",
                "headers": [(b"x-request-id", b"\xff")],
            },
            Mock(),
            send,
        )
    )
    UUID(dict(messages[0]["headers"])[b"x-request-id"].decode())
    assert (
        metrics.registry.get_sample_value(
            "credit_scoring_http_request_duration_seconds_sum",
            {"method": "GET", "route": "unmatched"},
        )
        == 0.25
    )


def test_incomplete_response_does_not_observe_probabilities() -> None:
    import asyncio

    from fastapi import FastAPI
    from starlette.types import Message, Receive, Scope, Send

    from credit_scoring.serving.metrics import Metrics
    from credit_scoring.serving.middleware import RequestMiddleware

    async def incomplete(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("incomplete-response-sentinel")

    async def send(message: Message) -> None:
        pass

    metrics = Metrics()
    middleware = RequestMiddleware(incomplete, metrics=metrics, logger=Mock(), key=None)
    asyncio.run(
        middleware(
            {
                "type": "http",
                "app": FastAPI(),
                "path": "/",
                "method": "POST",
                "headers": [],
                "state": {"score_probabilities": [0.5]},
            },
            Mock(),
            send,
        )
    )
    assert metrics.registry.get_sample_value("credit_scoring_probability_of_default_count") == 0
