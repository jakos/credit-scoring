from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from credit_scoring.registry.lifecycle import run_lifecycle, select_challenger
from credit_scoring.registry.policy import load_policy
from credit_scoring.training.train import LifecycleArtifacts

from .fakes import FakeRegistry


@pytest.fixture
def prepared(monkeypatch):
    import credit_scoring.registry.lifecycle as module

    frame = pd.DataFrame({"amount": [10.0, 20, 30, 40], "default.payment.next.month": [0, 0, 1, 1]})
    x = frame.drop(columns="default.payment.next.month")
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    estimator = SimpleNamespace(get_params=lambda: {"max_iter": 50})
    pipeline = SimpleNamespace(named_steps={"model": estimator})
    artifact = LifecycleArtifacts(
        pipeline, x, x, frame.iloc[:, 1], scores, scores, np.array([0, 0, 1, 1])
    )
    monkeypatch.setattr(
        module,
        "train_lifecycle",
        lambda df, max_iter: {"logistic_regression": artifact, "hist_gradient_boosting": artifact},
    )
    return frame, x


def champion(scores):
    pipeline = Mock()
    pipeline.classes_ = np.array([0, 1])
    pipeline.predict.return_value = (np.array(scores) >= 0.5).astype(int)
    pipeline.predict_proba.return_value = np.column_stack((1 - np.array(scores), scores))
    return pipeline


def test_bootstrap_and_cards_for_both_runs(prepared) -> None:
    frame, _ = prepared
    registry = FakeRegistry()
    assert run_lifecycle(frame, registry, load_policy()).approved
    assert registry.aliases == {"challenger": "2", "champion": "2"}
    assert len(registry.cards) == 2
    assert registry.tags["2"]["promotion_status"] == "promoted"
    for card in registry.cards.values():
        for required in [
            "sha256:",
            "default.payment.next.month",
            "split_seed",
            "train_size",
            "test_size",
            "Estimator parameters",
            "roc_auc",
            "pr_auc",
            "ks",
            "gini",
            "psi",
            "PSI methodology",
            "Policy",
            "Run/version lineage",
            "Limitations",
            "single writer",
            "fairness",
        ]:
            assert required in card
    assert '"status": "promoted"' in registry.cards["2"]


def test_champion_rescored_on_current_rows_and_promoted(prepared) -> None:
    frame, x = prepared
    old = champion([0.1, 0.8, 0.2, 0.9])
    registry = FakeRegistry(old)
    assert run_lifecycle(frame, registry, load_policy()).approved
    pd.testing.assert_frame_equal(old.predict.call_args.args[0], x)
    pd.testing.assert_frame_equal(old.predict_proba.call_args.args[0], x)
    assert registry.aliases["champion"] == "2"
    assert '"roc_auc": 0.75' in registry.cards["2"]


def test_rejection_preserves_prior_champion(prepared) -> None:
    frame, _ = prepared
    registry = FakeRegistry(champion([0.1, 0.2, 0.8, 0.9]))
    decision = run_lifecycle(frame, registry, load_policy())
    assert not decision.approved
    assert decision.reasons == ("roc_auc_improvement",)
    assert registry.assignments == [("challenger", "2")]
    assert registry.aliases["champion"] == "old"
    assert registry.tags["2"]["promotion_status"] == "rejected"
    assert "roc_auc_improvement" in registry.cards["2"]


@pytest.mark.parametrize("failure", ["incompatible", "concurrent", "alias", "logging", "loading"])
def test_failures_are_command_failures_and_never_promoted(prepared, failure) -> None:
    frame, _ = prepared
    registry = FakeRegistry(champion([0.1, 0.8, 0.2, 0.9]))
    if failure == "incompatible":
        registry.champion.predict_proba.side_effect = ValueError("incompatible features")
    elif failure == "concurrent":
        registry.change_champion = True
    elif failure == "alias":
        registry.fail_alias = "champion"
    elif failure == "logging":
        registry.log_model = Mock(side_effect=RuntimeError("registration unavailable"))
    else:
        registry.load_pipeline = Mock(side_effect=RuntimeError("loading unavailable"))
    with pytest.raises((RuntimeError, ValueError)):
        run_lifecycle(frame, registry, load_policy())
    assert registry.aliases["champion"] == "old"
    assert all(tags["promotion_status"] != "promoted" for tags in registry.tags.values())
    if failure != "logging":
        assert registry.tags["2"]["promotion_status"] == "failed"


@pytest.mark.parametrize(
    "a,b,expected",
    [((0.8, 0.7), (0.9, 0.6), "b"), ((0.8, 0.7), (0.8, 0.8), "b"), ((0.8, 0.7), (0.8, 0.7), "a")],
)
def test_selection_order(a, b, expected) -> None:
    evaluations = {
        name: dict(roc_auc=values[0], pr_auc=values[1], psi=0)
        for name, values in [("b", b), ("a", a)]
    }
    assert select_challenger(evaluations) == expected


def test_undefined_discrimination_prevents_selection() -> None:
    with pytest.raises(ValueError, match="Undefined"):
        select_challenger({"lr": {"roc_auc": float("nan"), "pr_auc": 0.7, "psi": 0.01}})


def test_incompatible_champion_class_order_fails_closed(prepared) -> None:
    frame, _ = prepared
    old = champion([0.1, 0.2, 0.8, 0.9])
    old.classes_ = np.array([1, 0])
    registry = FakeRegistry(old)
    with pytest.raises(ValueError, match="binary default classes"):
        run_lifecycle(frame, registry, load_policy())
    assert registry.aliases["champion"] == "old"
