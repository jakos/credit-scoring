"""Pure scoring orchestration; no configuration or model I/O."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from credit_scoring.serving.contracts import Explanation, ModelSnapshot
from credit_scoring.serving.schemas import FEATURE_MAX, FEATURE_MIN


@dataclass(frozen=True)
class ValidationIssue:
    loc: list[str | int]
    msg: str
    type: str


@dataclass(frozen=True)
class Score:
    probability_of_default: float
    risk_band: Literal["low", "medium", "high"]
    explanation: Explanation


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
                    if not finite or not FEATURE_MIN <= value <= FEATURE_MAX:
                        errors.append(
                            ValidationIssue(
                                loc=[*prefix, "features", name],
                                msg="Model feature must be between -1e12 and 1e12 inclusive",
                                type="feature_range",
                            )
                        )
            for position, name in enumerate(applicant):
                if name in contract.names:
                    continue
                # Numeric positions preserve distinct locations without disclosing keys.
                errors.append(
                    ValidationIssue(
                        loc=[*prefix, "features", position],
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
            if self.snapshot.explainer is None:
                raise PredictionError("Missing explanation component")
            explanations = self.snapshot.explainer.explain(frame)
            for explanation in explanations:
                _validate_explanation(explanation, contract.names)
            if len(explanations) != len(applicants):
                raise PredictionError("Invalid explanation count")
            return [
                Score(
                    probability_of_default=float(p),
                    explanation=explanation,
                    risk_band=("low" if p < self.medium else "medium" if p < self.high else "high"),
                )
                for p, explanation in zip(probabilities, explanations, strict=True)
            ]
        except Exception as exc:
            raise PredictionError("Model prediction failed") from exc


def _validate_explanation(explanation: Explanation, names: tuple[str, ...]) -> None:
    features = explanation.top_features
    if (
        explanation.method != "linear_log_odds"
        or len(features) != min(5, len(names))
        or len({f.feature for f in features}) != len(features)
        or any(f.feature not in names or not math.isfinite(f.contribution) for f in features)
        or not math.isfinite(explanation.baseline_log_odds)
        or not math.isfinite(explanation.remaining_contribution)
        or features != sorted(features, key=lambda f: (-abs(f.contribution), f.feature))
    ):
        raise PredictionError("Invalid explanation output")
