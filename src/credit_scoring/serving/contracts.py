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
class ModelSnapshot:
    predictor: Predictor
    features: FeatureContract
    default_column: int


class Loader(Protocol):
    def __call__(self, settings: ServingSettings) -> ModelSnapshot: ...
