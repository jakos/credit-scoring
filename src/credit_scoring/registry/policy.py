"""Reviewable, packaged promotion policy."""

import math
import tomllib
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class Policy:
    roc_auc_min_improvement: float
    pr_auc_max_decrease: float
    psi_ceiling: float
    bootstrap_roc_auc_min: float
    psi_bins: int
    psi_smoothing: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Invalid policy value: {field.name}")
            if field.name == "psi_bins":
                valid = isinstance(value, int) and 2 <= value <= 1000
            elif field.name in {"psi_ceiling", "psi_smoothing"}:
                valid = 0 < value <= 1
            else:
                valid = 0 <= value <= 1
            if not valid:
                raise ValueError(f"Invalid policy value: {field.name}")


def load_policy(path: Path | None = None) -> Policy:
    text = (
        path.read_text(encoding="utf-8")
        if path is not None
        else files(__package__).joinpath("promotion.toml").read_text(encoding="utf-8")
    )
    values = tomllib.loads(text)
    if set(values) != {field.name for field in fields(Policy)}:
        raise ValueError("Policy must contain exactly the declared policy keys")
    return Policy(**values)
