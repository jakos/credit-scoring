"""Aggregate metadata only: no raw observations or settings/credentials."""

import json
from dataclasses import asdict
from typing import Any

from credit_scoring.registry.contracts import ModelVersion
from credit_scoring.registry.policy import Policy

LIMITATIONS = "Held-out data is reused for selection and promotion; independent validation remains future work. These gates establish neither calibration, fairness nor regulatory approval. Promotion requires a single writer; alias updates are not transactional compare-and-swap."


def model_card(
    *,
    algorithm: str,
    dataset: dict[str, Any],
    parameters: dict[str, Any],
    metrics: dict[str, float],
    policy: Policy,
    model: ModelVersion,
    champion: str | None,
    outcome: str,
    reasons: tuple[str, ...] = (),
    champion_metrics: dict[str, float] | None = None,
) -> str:
    sections = {
        "Dataset and split": dataset,
        "Estimator parameters": parameters,
        "Metrics": metrics,
        "Policy": asdict(policy),
        "Run/version lineage": {**asdict(model), "prior_champion": champion},
        "Current champion metrics": champion_metrics,
        "Promotion": {"status": outcome, "reasons": reasons},
    }
    text = f"# Model card: {algorithm}\n\n"
    for heading, values in sections.items():
        text += f"## {heading}\n\n```json\n{json.dumps(values, indent=2, sort_keys=True, default=str)}\n```\n\n"
    return (
        text
        + "## PSI methodology\n\nTraining-derived quantile internal edges are deduplicated; infinite outer edges cover [0, 1]. Add policy smoothing to every bin proportion and renormalize, including empty bins. Constant reference scores retain one internal edge; identical constants give zero PSI.\n\n## Limitations\n\n"
        + LIMITATIONS
        + "\n"
    )
