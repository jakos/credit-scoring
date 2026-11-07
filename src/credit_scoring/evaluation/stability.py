"""Training-reference probability stability without storing observations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def probabilities(values: ArrayLike) -> NDArray[np.float64]:
    scores = np.asarray(values, dtype=float)
    if (
        scores.ndim != 1
        or scores.size == 0
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Expected nonempty finite one-dimensional probabilities in [0, 1]")
    return scores


def population_stability_index(
    reference: ArrayLike, observed: ArrayLike, *, bins: int, smoothing: float
) -> float:
    """Quantile bins, deduplicated internal edges, infinite outer coverage.

    Add smoothing to every bin proportion and renormalize. Constant reference
    scores retain their single internal edge, separating lower/higher shifts.
    Equal values always enter the same bin (numpy histogram convention).
    """
    ref, obs = probabilities(reference), probabilities(observed)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be positive and finite")
    internal = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    expected = np.histogram(ref, edges)[0] / ref.size + smoothing
    actual = np.histogram(obs, edges)[0] / obs.size + smoothing
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))
