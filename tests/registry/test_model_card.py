from credit_scoring.registry.contracts import ModelVersion
from credit_scoring.registry.model_card import model_card
from credit_scoring.registry.policy import load_policy


def test_card_is_markdown_with_aggregate_metadata() -> None:
    card = model_card(
        algorithm="logistic_regression",
        dataset={
            "identity": "sha256:abc",
            "target": "default",
            "features": {"amount": "float64"},
            "split_seed": 42,
            "train_size": 80,
            "test_size": 20,
        },
        parameters={"max_iter": 50},
        metrics={"roc_auc": 0.8, "pr_auc": 0.7, "ks": 0.5, "gini": 0.6, "psi": 0.02},
        policy=load_policy(),
        model=ModelVersion("run", "1", "models:/id"),
        champion=None,
        outcome="rejected",
        reasons=("bootstrap_roc_auc",),
    )
    assert card.startswith("# Model card: logistic_regression\n\n")
    assert "## PSI methodology\n" in card
    assert "0.005" in card and "0.01" in card and "0.1" in card
    assert "bootstrap_roc_auc" in card and "rejected" in card
    assert "calibration" in card and "regulatory approval" in card
    assert "independent validation" in card
