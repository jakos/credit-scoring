"""Pure promotion gates; no registry or process dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite

from credit_scoring.registry.policy import Policy


@dataclass(frozen=True)
class Decision:
    approved: bool
    gates: dict[str, bool]
    reasons: tuple[str, ...]


def decide(
    policy: Policy, challenger: Mapping[str, float], champion: Mapping[str, float] | None = None
) -> Decision:
    def metric(values: Mapping[str, float], key: str) -> Decimal | None:
        value = values.get(key)
        return Decimal(str(value)) if value is not None and isfinite(value) else None

    auc, pr, psi = (metric(challenger, key) for key in ("roc_auc", "pr_auc", "psi"))
    gates = {
        "required_metrics": auc is not None and pr is not None and psi is not None,
        "psi": psi is not None and psi < Decimal(str(policy.psi_ceiling)),
    }
    if champion is None:
        gates["bootstrap_roc_auc"] = auc is not None and auc >= Decimal(
            str(policy.bootstrap_roc_auc_min)
        )
    else:
        old_auc, old_pr = (metric(champion, key) for key in ("roc_auc", "pr_auc"))
        gates["roc_auc_improvement"] = (
            auc is not None
            and old_auc is not None
            and auc - old_auc >= Decimal(str(policy.roc_auc_min_improvement))
        )
        gates["pr_auc_decrease"] = (
            pr is not None
            and old_pr is not None
            and old_pr - pr <= Decimal(str(policy.pr_auc_max_decrease))
        )
    reasons = tuple(name for name, passed in gates.items() if not passed)
    return Decision(not reasons, gates, reasons)
