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
