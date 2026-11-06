"""Optional MLflow boundary; infrastructure failures propagate."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from credit_scoring.registry.contracts import ModelVersion


class MLflowRegistry:
    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        model_name: str,
        *,
        mlflow: Any = None,
        client: Any = None,
    ) -> None:
        self.mlflow = mlflow if mlflow is not None else import_module("mlflow")
        self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment)
        self.client = client if client is not None else self.mlflow.tracking.MlflowClient()
        self.model_name = model_name

    def log_model(
        self,
        algorithm: str,
        pipeline: Any,
        metrics: dict[str, float],
        tags: dict[str, str],
        card: Callable[[ModelVersion], str],
    ) -> ModelVersion:
        lineage = {**tags, "algorithm": algorithm}
        with self.mlflow.start_run(run_name=algorithm, tags=lineage) as run:
            # Write a complete pending card immediately, even if logging or registration fails.
            self.mlflow.log_text(
                card(ModelVersion(run.info.run_id, "unregistered", "unlogged")), "model_card.md"
            )
            self.mlflow.log_params(pipeline.named_steps["model"].get_params())
            self.mlflow.log_metrics(metrics)
            info = self.mlflow.sklearn.log_model(pipeline, name="model")
            version = self.mlflow.register_model(
                info.model_uri, self.model_name, await_registration_for=300, tags=lineage
            )
            ready = self.client.get_model_version(self.model_name, str(version.version))
            if ready.status != "READY":
                raise RuntimeError("Registered model version is not READY")
            return ModelVersion(run.info.run_id, str(version.version), info.model_uri)

    def resolve_alias(self, alias: str) -> str | None:
        try:
            registered_model = self.client.get_registered_model(self.model_name)
        except Exception as exc:
            # Only a missing registered model permits bootstrap after a lookup failure.
            # Never turn permissions, transport, or arbitrary errors into bootstrap.
            if getattr(exc, "error_code", None) == "RESOURCE_DOES_NOT_EXIST":
                return None
            raise
        version = registered_model.aliases.get(alias)
        return str(version) if version is not None else None

    def load_pipeline(self, version: str) -> Any:
        return self.mlflow.sklearn.load_model(f"models:/{self.model_name}/{version}")

    def assign_alias(self, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(self.model_name, alias, version)

    def record(self, model: ModelVersion, card: str, tags: dict[str, str]) -> None:
        with self.mlflow.start_run(run_id=model.run_id):
            self.mlflow.log_text(card, "model_card.md")
            self.mlflow.set_tags(tags)
        for key, value in tags.items():
            self.client.set_model_version_tag(self.model_name, model.version, key, value)
