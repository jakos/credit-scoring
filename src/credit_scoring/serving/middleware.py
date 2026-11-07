"""Request-local correlation and observations surrounding all application responses."""

import logging
import re
from collections.abc import Callable
from time import monotonic
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from credit_scoring.serving.auth import authorized
from credit_scoring.serving.metrics import Metrics

METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)
REQUEST_ID = re.compile(rb"[A-Za-z0-9._-]{1,128}")


class RequestMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: Metrics,
        logger: logging.Logger,
        key: bytes | None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.app, self.metrics, self.logger, self.key, self.clock = app, metrics, logger, key, clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = self.clock()
        ids = [v for k, v in scope["headers"] if k.lower() == b"x-request-id"]
        request_id = (
            ids[0].decode("ascii")
            if len(ids) == 1 and REQUEST_ID.fullmatch(ids[0])
            else str(uuid4())
        )
        method = scope["method"] if scope["method"] in METHODS else "OTHER"
        route = "unmatched"
        for candidate in scope["app"].routes:
            match, _ = candidate.matches(scope)
            if match != Match.NONE:
                route = candidate.path
                if match == Match.FULL:
                    break
        status = 500
        sent = False
        completed = False

        async def correlated(message: Message) -> None:
            nonlocal status, sent, completed
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ]
                message["headers"].append((b"x-request-id", request_id.encode("ascii")))
                sent = True
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                completed = True

        try:
            if authorized(scope, self.key):
                await self.app(scope, receive, correlated)
            else:
                await JSONResponse({"detail": "Unauthorized"}, status_code=401)(
                    scope, receive, correlated
                )
        except Exception:
            self.logger.error("serving_request_failed")
            completed = False
            if sent:
                raise
            if not sent:
                await JSONResponse({"detail": "Internal server error"}, status_code=500)(
                    scope, receive, correlated
                )
        finally:
            duration = max(0.0, self.clock() - started)
            if scope["path"] != "/metrics":
                self.metrics.requests.labels(method, route, str(status)).inc()
                self.metrics.duration.labels(method, route).observe(duration)
            if completed and status == 200:
                for probability in scope.get("state", {}).get("score_probabilities", []):
                    self.metrics.probability.observe(probability)
            self.logger.info(
                "serving_access",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": method,
                        "route": route,
                        "status": status,
                        "duration_seconds": duration,
                    }
                },
            )
