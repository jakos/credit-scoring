"""Small registry boundary used by lifecycle orchestration and offline fakes."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelVersion:
    run_id: str
    version: str
    model_uri: str


class Registry(Protocol):
    def log_model(
        self,
        algorithm: str,
        pipeline: Any,
        metrics: dict[str, float],
        tags: dict[str, str],
        card: Callable[[ModelVersion], str],
    ) -> ModelVersion: ...
    def resolve_alias(self, alias: str) -> str | None: ...
    def load_pipeline(self, version: str) -> Any: ...
    def assign_alias(self, alias: str, version: str) -> None: ...
    def record(self, model: ModelVersion, card: str, tags: dict[str, str]) -> None: ...
