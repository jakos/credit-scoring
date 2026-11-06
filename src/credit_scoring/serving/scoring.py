"""Pure scoring orchestration; no configuration or model I/O."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from credit_scoring.serving.contracts import ModelSnapshot


@dataclass(frozen=True)
class ValidationIssue:
    loc: list[str | int]
    msg: str
    type: str


@dataclass(frozen=True)
class Score:
    probability_of_default: float
    risk_band: Literal["low", "medium", "high"]


class FeatureValidationError(ValueError):
    def __init__(self, details: list[ValidationIssue]) -> None:
        super().__init__("Applicant features do not match the model")
        self.details = details


class PredictionError(RuntimeError):
    """The predictor failed or returned invalid probabilities."""


@dataclass(frozen=True)
class Scorer:
    snapshot: ModelSnapshot
    medium: float
    high: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.medium)
            and math.isfinite(self.high)
            and 0 < self.medium < self.high < 1
        ):
            raise ValueError("Require finite thresholds: 0 < medium < high < 1")

    def score(
        self, applicants: Sequence[Mapping[str, float | int | None]], *, batch: bool = False
    ) -> list[Score]:
        errors: list[ValidationIssue] = []
        contract = self.snapshot.features
        for index, applicant in enumerate(applicants):
            prefix: list[str | int] = ["body", "applicants", index] if batch else ["body"]
            for name in contract.names:
                if name not in applicant:
                    errors.append(
                        ValidationIssue(
                            loc=[*prefix, "features", name],
                            msg="Required model feature is missing",
                            type="missing",
                        )
                    )
                elif applicant[name] is None and name not in contract.nullable:
                    errors.append(
                        ValidationIssue(
                            loc=[*prefix, "features", name],
                            msg="Model does not support null for this feature",
                            type="null_not_supported",
                        )
                    )
                elif (value := applicant[name]) is not None:
                    try:
                        finite = math.isfinite(float(value))
                    except OverflowError:
                        finite = False
                    if not finite:
                        errors.append(
                            ValidationIssue(
                                loc=[*prefix, "features", name],
                                msg="Model feature must be representable as a finite float",
                                type="finite_number",
                            )
                        )
            for _name in applicant.keys() - set(contract.names):
                # Unknown keys can contain sensitive values; report the envelope only.
                errors.append(
                    ValidationIssue(
                        loc=[*prefix, "features"],
                        msg="Unknown model feature",
                        type="extra_forbidden",
                    )
                )
        if errors:
            raise FeatureValidationError(errors)
        frame = pd.DataFrame(applicants, columns=list(contract.names), dtype=float)
        try:
            output = np.asarray(self.snapshot.predictor.predict_proba(frame), dtype=float)
            if output.shape != (len(applicants), 2):
                raise PredictionError("Invalid probability shape")
            if not np.isfinite(output).all() or ((output < 0) | (output > 1)).any():
                raise PredictionError("Invalid probability values")
            probabilities = output[:, self.snapshot.default_column]
            return [
                Score(
                    probability_of_default=float(p),
                    risk_band=("low" if p < self.medium else "medium" if p < self.high else "high"),
                )
                for p in probabilities
            ]
        except Exception:
            raise PredictionError("Model prediction failed") from None
