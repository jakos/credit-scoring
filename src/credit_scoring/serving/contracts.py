"""Small prediction and loading boundaries shared by scoring and startup."""

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from credit_scoring.config.settings import ServingSettings


class Predictor(Protocol):
    def predict_proba(self, frame: pd.DataFrame) -> Any: ...


@dataclass(frozen=True)
class FeatureContract:
    names: tuple[str, ...]
    nullable: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    contribution: float


@dataclass(frozen=True)
class Explanation:
    method: str
    baseline_log_odds: float
    top_features: list[FeatureContribution]
    remaining_contribution: float


class Explainer(Protocol):
    def explain(self, frame: pd.DataFrame) -> list[Explanation]: ...


@dataclass(frozen=True)
class ModelSnapshot:
    predictor: Predictor
    features: FeatureContract
    default_column: int
    explainer: Explainer | None = None


class Loader(Protocol):
    def __call__(self, settings: ServingSettings) -> ModelSnapshot: ...
