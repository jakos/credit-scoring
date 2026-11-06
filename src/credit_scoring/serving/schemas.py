"""Validated HTTP boundaries and a synthetic UCI-shaped applicant example."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

FeatureValue = Annotated[StrictFloat, Field(allow_inf_nan=False)] | StrictInt | None

# Synthetic values, using the numeric feature names selected by default training.
APPLICANT_EXAMPLE: dict[str, Any] = {
    "features": {
        "LIMIT_BAL": 50000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 10000,
        "BILL_AMT2": 9000,
        "BILL_AMT3": 8000,
        "BILL_AMT4": 7000,
        "BILL_AMT5": 6000,
        "BILL_AMT6": 5000,
        "PAY_AMT1": 1000,
        "PAY_AMT2": 1000,
        "PAY_AMT3": 1000,
        "PAY_AMT4": 1000,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 1000,
    }
}


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Applicant(Envelope):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [APPLICANT_EXAMPLE]})
    features: dict[str, FeatureValue]


class BatchRequest(Envelope):
    applicants: list[Applicant] = Field(min_length=1, max_length=1000)


class ScoreResponse(Envelope):
    probability_of_default: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk_band: Literal["low", "medium", "high"]


class BatchResponse(Envelope):
    scores: list[ScoreResponse]


class HealthResponse(Envelope):
    status: Literal["live", "ready", "unready"]


class ErrorDetail(Envelope):
    loc: list[str | int]
    msg: str
    type: str


class ErrorResponse(Envelope):
    detail: str | list[ErrorDetail]
