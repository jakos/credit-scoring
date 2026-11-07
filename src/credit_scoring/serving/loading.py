"""Resolve and validate a trusted model once, outside request handling."""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from credit_scoring.config.settings import ServingSettings
from credit_scoring.registry.contracts import Registry
from credit_scoring.registry.mlflow_adapter import MLflowRegistry
from credit_scoring.serving.contracts import FeatureContract, ModelSnapshot
from credit_scoring.serving.explanations import build_explanation


class RegistryLoadError(RuntimeError):
    """Registry access failed while resolving or loading the champion."""


class ModelContractError(ValueError):
    """Artifact metadata cannot establish a compatible serving contract."""


def _nullable_features(pipeline: Any, names: tuple[str, ...]) -> frozenset[str]:
    """Recognize only the inherited fitted numeric preprocessing contract."""
    if not isinstance(pipeline, Pipeline):
        return frozenset()
    preprocessor = pipeline.named_steps.get("preprocessor")
    if not isinstance(preprocessor, ColumnTransformer):
        return frozenset()
    nullable: set[str] = set()
    for _, transform, columns in preprocessor.transformers_:
        if not isinstance(transform, Pipeline):
            continue
        imputer = transform.steps[0][1]
        if not isinstance(imputer, SimpleImputer) or imputer.strategy != "median":
            continue
        if not isinstance(columns, (list, tuple)) or not all(isinstance(c, str) for c in columns):
            continue
        if len(columns) != len(imputer.statistics_):
            continue
        nullable.update(
            c
            for c, statistic in zip(columns, imputer.statistics_, strict=True)
            if np.isfinite(statistic) and c in names
        )
    return frozenset(nullable)


def validate_pipeline(pipeline: Any) -> ModelSnapshot:
    names = getattr(pipeline, "feature_names_in_", None)
    if names is None or np.asarray(names).ndim != 1 or len(names) == 0:
        raise ModelContractError("Required input feature metadata is missing")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ModelContractError("Feature names must be nonempty strings")
    names = tuple(str(name) for name in names)
    if len(set(names)) != len(names) or {"ID", "default.payment.next.month"}.intersection(names):
        raise ModelContractError("Incompatible input feature metadata")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ModelContractError("Model must support predict_proba")
    classes = np.asarray(getattr(pipeline, "classes_", []))
    if classes.shape != (2,) or classes.dtype.kind not in "iuf" or not np.isfinite(classes).all():
        raise ModelContractError("Binary numeric class metadata is required")
    indexes = np.flatnonzero(classes == 1)
    if len(indexes) != 1 or classes[0] == classes[1]:
        raise ModelContractError("Binary classes must include default label 1 exactly once")
    try:
        explainer = build_explanation(pipeline, names, int(indexes[0]))
    except Exception as exc:
        raise ModelContractError("Unsupported explanation contract") from exc
    return ModelSnapshot(
        pipeline,
        FeatureContract(names, _nullable_features(pipeline, names)),
        int(indexes[0]),
        explainer,
    )


def load_snapshot(settings: ServingSettings, *, registry: Registry | None = None) -> ModelSnapshot:
    if settings.serving_model_source == "local":
        from mlflow.sklearn import load_model

        pipeline = load_model(str(settings.serving_model_path))
    else:
        try:
            if registry is None:
                if settings.mlflow_tracking_uri is None:
                    raise ValueError("Registry serving requires a tracking URI")
                registry = MLflowRegistry(
                    settings.mlflow_tracking_uri,
                    settings.mlflow_experiment,
                    settings.registered_model_name,
                )
            version = registry.resolve_alias("champion")
            pipeline = registry.load_pipeline(version) if version is not None else None
        except Exception as exc:
            raise RegistryLoadError("Registry lookup or model loading failed") from exc
        if version is None:
            raise ModelContractError("No champion version is available")
    return validate_pipeline(pipeline)
