from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from credit_scoring.config.settings import ServingSettings
from credit_scoring.serving.app import create_app
from credit_scoring.serving.contracts import ModelSnapshot
from credit_scoring.serving.loading import ModelContractError, validate_pipeline
from credit_scoring.serving.schemas import APPLICANT_EXAMPLE
from credit_scoring.serving.scoring import Scorer


@pytest.mark.parametrize("reverse", [False, True])
def test_native_contributions_and_reconstruction(pipeline: Pipeline, reverse: bool) -> None:
    pipeline = deepcopy(pipeline)
    if reverse:
        pipeline[-1].classes_ = np.array([1, 0])
    snapshot = validate_pipeline(pipeline)
    features = {**APPLICANT_EXAMPLE["features"], "AGE": None}
    shuffled = dict(reversed(list(features.items())))
    frame = pd.DataFrame([shuffled], columns=snapshot.features.names, dtype=float)
    explanation = snapshot.explainer.explain(frame)[0]
    orientation = -1 if reverse else 1
    values = pipeline[:-1].transform(frame)[0] * pipeline[-1].coef_[0] * orientation
    expected = sorted(
        zip(snapshot.explainer.names, values, strict=True),
        key=lambda pair: (-abs(pair[1]), pair[0]),
    )
    assert [(f.feature, f.contribution) for f in explanation.top_features] == expected[:5]
    assert explanation.remaining_contribution == pytest.approx(sum(v for _, v in expected[5:]))
    total = (
        explanation.baseline_log_odds
        + sum(f.contribution for f in explanation.top_features)
        + explanation.remaining_contribution
    )
    assert total == pytest.approx(orientation * pipeline.decision_function(frame)[0])
    assert explanation.baseline_log_odds == orientation * pipeline[-1].intercept_[0]
    assert any(v < 0 for v in values) and any(v > 0 for v in values)
    scorer = Scorer(snapshot, 0.2, 0.5)
    assert scorer.score([features]) == scorer.score([shuffled])


def test_ties_and_fewer_than_five() -> None:
    from sklearn.linear_model import LogisticRegression

    from credit_scoring.features.preprocess import build_preprocessor

    frame = pd.DataFrame({"z": [-1.0, 1.0, -1.0, 1.0], "a": [-1.0, 1.0, -1.0, 1.0]})
    pipeline = Pipeline(
        [("preprocessor", build_preprocessor(["z", "a"])), ("model", LogisticRegression())]
    ).fit(frame, [0, 1, 0, 1])
    pipeline[-1].coef_[:] = 1
    explanation = validate_pipeline(pipeline).explainer.explain(frame.iloc[[1]])[0]
    assert [f.feature for f in explanation.top_features] == ["a", "z"]
    assert explanation.remaining_contribution == 0


@pytest.mark.parametrize("mutation", ["mapping", "model", "parameters"])
def test_unsupported_contract_is_unready(
    pipeline: Pipeline, settings: ServingSettings, mutation: str
) -> None:
    pipeline = deepcopy(pipeline)
    if mutation == "mapping":
        pipeline[0].transformers_[0][1].steps[0][1].add_indicator = True
    elif mutation == "model":
        pipeline.steps[-1] = ("model", Mock(classes_=np.array([0, 1])))
    else:
        pipeline[-1].coef_[0, 0] = np.nan
    with pytest.raises(ModelContractError):
        validate_pipeline(pipeline)
    with TestClient(create_app(settings, lambda _: validate_pipeline(pipeline))) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert client.post("/v1/score", json=APPLICANT_EXAMPLE).status_code == 503


@pytest.mark.parametrize("output", [[], [object()], [float("nan")]])
def test_runtime_explanation_failure_is_atomic(
    settings: ServingSettings, snapshot: ModelSnapshot, output: list[object]
) -> None:
    explainer = Mock()
    explainer.explain.return_value = output
    app = create_app(settings, lambda _: replace(snapshot, explainer=explainer))
    with TestClient(app) as client:
        for path, body in [
            ("/v1/score", APPLICANT_EXAMPLE),
            ("/v1/score/batch", {"applicants": [APPLICANT_EXAMPLE] * 2}),
        ]:
            response = client.post(path, json=body)
            assert response.status_code == 500
            assert response.json() == {"detail": "Model prediction failed"}
        assert (
            app.state.metrics.registry.get_sample_value(
                "credit_scoring_probability_of_default_count"
            )
            == 0
        )


def test_decision_disagreement_is_rejected(pipeline: Pipeline) -> None:
    snapshot = validate_pipeline(deepcopy(pipeline))
    snapshot.explainer.pipeline[-1].decision_function = Mock(return_value=np.array([123456.0]))
    with pytest.raises(ValueError, match="explanation"):
        snapshot.explainer.explain(pd.DataFrame([APPLICANT_EXAMPLE["features"]]))


def test_duplicate_transformed_mapping_rejected(pipeline: Pipeline) -> None:
    pipeline = deepcopy(pipeline)
    name, transform, columns = pipeline[0].transformers_[0]
    pipeline[0].transformers_.append((name + "_duplicate", transform, columns))
    with pytest.raises(ModelContractError, match="explanation"):
        validate_pipeline(pipeline)


def test_nonfinite_runtime_contributions_rejected(pipeline: Pipeline) -> None:
    pipeline = deepcopy(pipeline)
    snapshot = validate_pipeline(pipeline)
    snapshot.explainer.coefficients[0] = np.nan
    with pytest.raises(ValueError, match="explanation"):
        snapshot.explainer.explain(pd.DataFrame([APPLICANT_EXAMPLE["features"]]))


def test_mapping_uses_transformed_column_metadata(pipeline: Pipeline) -> None:
    from credit_scoring.features.preprocess import build_preprocessor

    names = list(pipeline.feature_names_in_)
    frame = pd.DataFrame(np.random.default_rng(12).normal(size=(40, len(names))), columns=names)
    pipeline = deepcopy(pipeline)
    pipeline.steps[0] = ("preprocessor", build_preprocessor(list(reversed(names))))
    pipeline.fit(frame, [0, 1] * 20)
    snapshot = validate_pipeline(pipeline)
    result = snapshot.explainer.explain(frame.iloc[[0]])[0]
    expected = dict(
        zip(
            reversed(names),
            pipeline[0].transform(frame.iloc[[0]])[0] * pipeline[-1].coef_[0],
            strict=True,
        )
    )
    assert all(f.contribution == pytest.approx(expected[f.feature]) for f in result.top_features)


def test_invalid_native_explanation_is_sanitized(
    settings: ServingSettings, snapshot: ModelSnapshot
) -> None:
    from credit_scoring.serving.contracts import Explanation, FeatureContribution

    invalid = Explanation(
        "linear_log_odds",
        float("nan"),
        [FeatureContribution(name, 0.0) for name in sorted(snapshot.features.names)[:5]],
        0.0,
    )
    explainer = Mock()
    explainer.explain.return_value = [invalid]
    app = create_app(settings, lambda _: replace(snapshot, explainer=explainer))
    with TestClient(app) as client:
        response = client.post("/v1/score", json=APPLICANT_EXAMPLE)
        assert response.status_code == 500
        assert response.json() == {"detail": "Model prediction failed"}
        assert (
            app.state.metrics.registry.get_sample_value(
                "credit_scoring_probability_of_default_count"
            )
            == 0
        )
