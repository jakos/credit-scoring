from credit_scoring import score_applicant


def test_score_applicant_high_income_low_debt() -> None:
    assert score_applicant(income=10000, debt=1000) == 0.9


def test_score_applicant_non_positive_income() -> None:
    assert score_applicant(income=0, debt=1000) == 0.0
