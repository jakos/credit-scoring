import pytest
from pydantic import ValidationError

from credit_scoring.serving.schemas import APPLICANT_EXAMPLE, Applicant, BatchRequest


@pytest.mark.parametrize(
    "value", ["2", True, False, float("inf"), float("-inf"), float("nan"), [], {}]
)
def test_invalid_feature_values(value: object) -> None:
    with pytest.raises(ValidationError):
        Applicant.model_validate({"features": {"AGE": value}})


@pytest.mark.parametrize("value", [0, 42, -1.5, 1.1, None])
def test_finite_numbers_and_explicit_null(value: object) -> None:
    assert Applicant.model_validate({"features": {"AGE": value}}).features["AGE"] == value


@pytest.mark.parametrize(
    "payload",
    [
        {"features": {}, "ID": 1},
        {"applicants": [APPLICANT_EXAMPLE], "extra": 1},
        {"applicants": []},
        {"applicants": [APPLICANT_EXAMPLE] * 1001},
        {"applicants": [APPLICANT_EXAMPLE, {"features": {"AGE": "secret"}}]},
    ],
)
def test_envelopes_and_batch_limits(payload: dict[str, object]) -> None:
    schema = BatchRequest if "applicants" in payload else Applicant
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


def test_example_and_maximum_batch() -> None:
    assert Applicant.model_validate(APPLICANT_EXAMPLE).features["AGE"] == 35
    assert (
        len(BatchRequest.model_validate({"applicants": [APPLICANT_EXAMPLE] * 1000}).applicants)
        == 1000
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["baseline_log_odds", "remaining_contribution", "contribution"])
def test_explanation_numbers_are_finite(field: str, value: float) -> None:
    from copy import deepcopy

    from credit_scoring.serving.schemas import SCORE_EXAMPLE, ScoreResponse

    response = deepcopy(SCORE_EXAMPLE)
    explanation = response["explanation"]
    if field == "contribution":
        explanation["top_features"][0][field] = value
    else:
        explanation[field] = value
    with pytest.raises(ValidationError):
        ScoreResponse.model_validate(response)


def test_explanation_required_and_openapi_example() -> None:
    from credit_scoring.serving.schemas import SCORE_EXAMPLE, ScoreResponse

    assert ScoreResponse.model_validate(SCORE_EXAMPLE).explanation.method == "linear_log_odds"
    with pytest.raises(ValidationError):
        ScoreResponse.model_validate({"probability_of_default": 0.5, "risk_band": "high"})


@pytest.mark.parametrize("value", [1e308, -1e308, 10**400, -(10**400), 1e12 + 1, -1e12 - 1])
def test_numeric_bound_rejects_extreme_finite_inputs(value: float | int) -> None:
    with pytest.raises(ValidationError):
        Applicant.model_validate({"features": {"AGE": value}})


@pytest.mark.parametrize("value", [-1e12, 1e12, -(10**12), 10**12])
def test_numeric_bound_is_inclusive(value: float | int) -> None:
    assert Applicant.model_validate({"features": {"AGE": value}}).features["AGE"] == value
