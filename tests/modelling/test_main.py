import argparse
import sys
from types import SimpleNamespace

from credit_scoring.modeling import main as modeling_main


def test_parse_args_uses_default_max_iter(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])

    args = modeling_main.parse_args()

    assert args.max_iter == 1000


def test_parse_args_reads_custom_max_iter(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--max-iter", "250"])

    args = modeling_main.parse_args()

    assert args.max_iter == 250


def test_main_trains_evaluates_and_prints_metrics(monkeypatch, capsys) -> None:
    fake_args = argparse.Namespace(max_iter=77)
    called: dict[str, int | None] = {"max_iter": None}

    def fake_train(*, max_iter: int):
        called["max_iter"] = max_iter
        return SimpleNamespace(
            y_test=[0, 1, 1, 0],
            y_pred=[0, 1, 0, 0],
            y_proba=[0.1, 0.8, 0.4, 0.3],
        )

    fake_metrics = {
        "accuracy": 0.75,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "roc_auc": 0.85,
        "pr_auc": 0.80,
    }

    monkeypatch.setattr(modeling_main, "parse_args", lambda: fake_args)
    monkeypatch.setattr(modeling_main, "train", fake_train)
    monkeypatch.setattr(modeling_main, "evaluate_predictions", lambda **_: fake_metrics)

    modeling_main.main()

    output = capsys.readouterr().out
    assert called["max_iter"] == 77
    assert "Model training completed." in output
    assert "Test samples: 4" in output
    assert "- accuracy: 0.7500" in output
    assert "- precision: 1.0000" in output
    assert "- recall: 0.5000" in output
    assert "- f1: 0.6667" in output
    assert "- roc_auc: 0.8500" in output
    assert "- pr_auc: 0.8000" in output
