"""Application factory with an isolated, immutable startup model snapshot."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from credit_scoring.config.settings import ServingSettings
from credit_scoring.logging import configure_logging
from credit_scoring.serving.contracts import Loader
from credit_scoring.serving.loading import load_snapshot
from credit_scoring.serving.metrics import Metrics
from credit_scoring.serving.middleware import RequestMiddleware
from credit_scoring.serving.routes import create_router
from credit_scoring.serving.schemas import ErrorDetail
from credit_scoring.serving.scoring import FeatureValidationError, PredictionError, Scorer


def create_app(settings: ServingSettings | None = None, loader: Loader = load_snapshot) -> FastAPI:
    configuration = settings if settings is not None else ServingSettings()
    logger = configure_logging(configuration.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.scorer = None
        try:
            snapshot = await run_in_threadpool(loader, configuration)
            app.state.scorer = Scorer(
                snapshot, configuration.risk_medium_threshold, configuration.risk_high_threshold
            )
        except Exception:
            # Exception text and tracebacks may contain credentials, paths or records.
            logger.error("serving_model_load_failed")
        try:
            yield
        finally:
            app.state.scorer = None

    app = FastAPI(title="Credit-risk scoring API", version="1.0.0", lifespan=lifespan)
    app.state.scorer = None
    app.include_router(create_router())
    metrics = Metrics()
    app.state.metrics = metrics
    key = configuration.serving_api_key
    app.add_middleware(
        RequestMiddleware,
        metrics=metrics,
        logger=logger,
        key=key.get_secret_value().encode("utf-8") if key is not None else None,
    )

    @app.get("/metrics", include_in_schema=False)
    def exposition() -> Response:
        return Response(
            generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    if key is not None:
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["APIKey"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
        for path in ("/v1/score", "/v1/score/batch"):
            schema["paths"][path]["post"]["security"] = [{"APIKey": []}]
            schema["paths"][path]["post"]["responses"]["401"] = {"description": "Unauthorized"}
        app.openapi_schema = schema

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for error in exc.errors():
            location = list(error["loc"])
            if error["type"] == "extra_forbidden":
                location = location[:-1]
            details.append({"loc": location, "msg": error["msg"], "type": error["type"]})
        return JSONResponse(status_code=422, content={"detail": details})

    @app.exception_handler(FeatureValidationError)
    async def invalid_features(request: Request, exc: FeatureValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    ErrorDetail.model_validate(d, from_attributes=True).model_dump()
                    for d in exc.details
                ]
            },
        )

    @app.exception_handler(PredictionError)
    async def prediction_failed(request: Request, exc: PredictionError) -> JSONResponse:
        logger.error("serving_prediction_failed")
        return JSONResponse(status_code=500, content={"detail": "Model prediction failed"})

    return app
