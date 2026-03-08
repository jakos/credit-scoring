"""Credit scoring package."""

__all__ = ["score_applicant"]


def score_applicant(income: float, debt: float) -> float:
    """Return a simple credit score in range [0.0, 1.0]."""
    if income <= 0:
        return 0.0

    debt_ratio = max(debt / income, 0.0)
    score = 1.0 - min(debt_ratio, 1.0)
    return round(score, 4)
