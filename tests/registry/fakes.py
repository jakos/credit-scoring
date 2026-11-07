"""Offline lifecycle registry with observable calls and failure injection."""

from typing import Any

from credit_scoring.registry.contracts import ModelVersion


class FakeRegistry:
    def __init__(self, champion: Any = None) -> None:
        self.aliases = {} if champion is None else {"champion": "old"}
        self.champion = champion
        self.models = []
        self.cards = {}
        self.tags = {}
        self.assignments = []
        self.change_champion = False
        self.fail_alias = None

    def log_model(self, algorithm, pipeline, metrics, tags, card) -> ModelVersion:
        model = ModelVersion(
            f"run-{algorithm}", str(len(self.models) + 1), f"models:/logged-{algorithm}"
        )
        self.models.append((algorithm, pipeline, metrics, tags))
        return model

    def resolve_alias(self, alias) -> str | None:
        if self.change_champion and self.models and alias == "champion":
            return "concurrent"
        return self.aliases.get(alias)

    def load_pipeline(self, version) -> Any:
        assert version == "old"
        return self.champion

    def assign_alias(self, alias, version) -> None:
        if alias == self.fail_alias:
            raise RuntimeError("alias unavailable")
        self.assignments.append((alias, version))
        self.aliases[alias] = version

    def record(self, model, card, tags) -> None:
        self.cards[model.version] = card
        self.tags[model.version] = tags
