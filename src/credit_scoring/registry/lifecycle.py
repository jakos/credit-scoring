"""Shared lifecycle orchestration; registry implementations are injected."""

import hashlib
import json
import math
from contextlib import suppress
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from credit_scoring.evaluation.metrics import evaluate_predictions
from credit_scoring.evaluation.stability import population_stability_index
from credit_scoring.features.preprocess import TARGET_COLUMN
from credit_scoring.registry.contracts import ModelVersion, Registry
from credit_scoring.registry.model_card import model_card
from credit_scoring.registry.policy import Policy
from credit_scoring.registry.promotion import Decision, decide
from credit_scoring.training.train import train_lifecycle


def select_challenger(evaluations: dict[str, dict[str, float]]) -> str:
    """Highest AUC, then AP, then lexicographically smallest algorithm name."""
    for metrics in evaluations.values():
        if not all(math.isfinite(metrics[key]) for key in ("roc_auc", "pr_auc", "psi")):
            raise ValueError("Undefined model discrimination or stability")
    return min(
        evaluations,
        key=lambda name: (-evaluations[name]["roc_auc"], -evaluations[name]["pr_auc"], name),
    )


def run_lifecycle(
    df: pd.DataFrame, registry: Registry, policy: Policy, *, max_iter: int = 1000
) -> Decision:
    expected_champion = registry.resolve_alias("champion")
    artifacts = train_lifecycle(df, max_iter)
    evaluations = {}
    for name, artifact in artifacts.items():
        metrics = evaluate_predictions(
            artifact.y_test, pd.Series(artifact.predictions), pd.Series(artifact.test_scores)
        )
        metrics["psi"] = population_stability_index(
            artifact.train_scores,
            artifact.test_scores,
            bins=policy.psi_bins,
            smoothing=policy.psi_smoothing,
        )
        evaluations[name] = metrics
    selected = select_challenger(evaluations)
    current = artifacts[selected]
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes()
        + json.dumps([(str(k), str(v)) for k, v in df.dtypes.items()]).encode()
    ).hexdigest()
    dataset: dict[str, Any] = {
        "identity": "sha256:" + fingerprint,
        "target": TARGET_COLUMN,
        "features": {str(key): str(value) for key, value in current.x_train.dtypes.items()},
        "split_seed": 42,
        "train_size": len(current.x_train),
        "test_size": len(current.x_test),
    }
    versions: dict[str, ModelVersion] = {}
    outcomes: dict[str, str] = {}
    reasons: tuple[str, ...] = ()
    champion_metrics = None

    def card(name: str, model: ModelVersion) -> str:
        artifact = artifacts[name]
        return model_card(
            algorithm=name,
            dataset=dataset,
            parameters=artifact.pipeline.named_steps["model"].get_params(),
            metrics=evaluations[name],
            policy=policy,
            model=model,
            champion=expected_champion,
            outcome=outcomes[name],
            reasons=reasons if name == selected else (),
            champion_metrics=champion_metrics,
        )

    def record(name: str) -> None:
        registry.record(
            versions[name],
            card(name, versions[name]),
            {
                "promotion_status": outcomes[name],
                "promotion_reasons": ", ".join(reasons) if name == selected else "",
                "selected_challenger": str(name == selected).lower(),
            },
        )

    try:
        for name, artifact in artifacts.items():
            outcomes[name] = "pending" if name == selected else "not_selected"
            versions[name] = registry.log_model(
                name,
                artifact.pipeline,
                evaluations[name],
                {"dataset_fingerprint": fingerprint, "prior_champion": expected_champion or "none"},
                partial(card, name),
            )
            record(name)
        registry.assign_alias("challenger", versions[selected].version)
        if expected_champion is not None:
            champion = registry.load_pipeline(expected_champion)
            if not np.array_equal(champion.classes_, [0, 1]):
                raise ValueError("Champion must predict binary default classes [0, 1]")
            champion_metrics = evaluate_predictions(
                current.y_test,
                pd.Series(champion.predict(current.x_test)),
                pd.Series(champion.predict_proba(current.x_test)[:, 1]),
            )
        decision = decide(policy, evaluations[selected], champion_metrics)
        reasons = decision.reasons
        if decision.approved:
            if registry.resolve_alias("champion") != expected_champion:
                raise RuntimeError("Champion changed during evaluation; refusing promotion")
            registry.assign_alias("champion", versions[selected].version)
            outcomes[selected] = "promoted"
        else:
            outcomes[selected] = "rejected"
        record(selected)
        return decision
    except Exception:
        # Never claim successful assignment when any registry operation failed.
        # Initial cards already exist before alias mutation; record failure where possible.
        if selected in versions and outcomes.get(selected) != "promoted":
            outcomes[selected] = "failed"
            reasons = ("lifecycle_failed",)
            # Preserve the original infrastructure/prediction failure.
            with suppress(Exception):
                record(selected)
        raise
