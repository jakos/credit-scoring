"""Thin synchronous HTTP routes; sklearn inference runs in FastAPI's thread pool."""

from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from credit_scoring.serving.schemas import (
    Applicant,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    HealthResponse,
    ScoreResponse,
)
from credit_scoring.serving.scoring import Scorer


def get_scorer(request: Request) -> Scorer:
    scorer = cast(Scorer | None, request.app.state.scorer)
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return scorer


def create_router() -> APIRouter:
    router = APIRouter()
    errors: dict[int | str, dict[str, Any]] = {
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    }

    @router.get("/healthz", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="live")

    @router.get(
        "/readyz", response_model=HealthResponse, responses={503: {"model": HealthResponse}}
    )
    def readiness(request: Request) -> HealthResponse | JSONResponse:
        if request.app.state.scorer is None:
            return JSONResponse(status_code=503, content={"status": "unready"})
        return HealthResponse(status="ready")

    @router.post("/v1/score", response_model=ScoreResponse, responses=errors)
    def score(
        applicant: Applicant, scorer: Annotated[Scorer, Depends(get_scorer)]
    ) -> ScoreResponse:
        result = scorer.score([applicant.features])[0]
        return ScoreResponse.model_validate(result, from_attributes=True)

    @router.post("/v1/score/batch", response_model=BatchResponse, responses=errors)
    def score_batch(
        batch: BatchRequest, scorer: Annotated[Scorer, Depends(get_scorer)]
    ) -> BatchResponse:
        results = scorer.score([applicant.features for applicant in batch.applicants], batch=True)
        return BatchResponse(
            scores=[
                ScoreResponse.model_validate(result, from_attributes=True) for result in results
            ]
        )

    return router
