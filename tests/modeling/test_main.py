import argparse
import sys
from types import SimpleNamespace

from credit_scoring.modeling import main as modeling_main


def test_parse_args_uses_default_max_iter(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])

    args = modeling_main.parse_args()

    assert args.max_iter == 1000
    assert args.enable_mlflow is False
    assert args.mlflow_tracking_uri is None
    assert args.mlflow_experiment == "credit-scoring"
    assert args.mlflow_run_name is None


def test_parse_args_reads_custom_max_iter(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--max-iter",
            "250",
            "--enable-mlflow",
            "--mlflow-tracking-uri",
            "http://127.0.0.1:5000",
            "--mlflow-experiment",
            "demo-exp",
            "--mlflow-run-name",
            "baseline",
        ],
    )

    args = modeling_main.parse_args()

    assert args.max_iter == 250
    assert args.enable_mlflow is True
    assert args.mlflow_tracking_uri == "http://127.0.0.1:5000"
    assert args.mlflow_experiment == "demo-exp"
    assert args.mlflow_run_name == "baseline"


def test_main_trains_evaluates_and_prints_metrics(monkeypatch, capsys) -> None:
    fake_args = argparse.Namespace(
        max_iter=77,
        enable_mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_experiment="credit-scoring",
        mlflow_run_name=None,
    )
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


def test_main_logs_to_mlflow_when_enabled(monkeypatch, capsys) -> None:
    fake_args = argparse.Namespace(
        max_iter=33,
        enable_mlflow=True,
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment="credit-scoring-exp",
        mlflow_run_name="run-1",
    )
    calls: dict[str, object] = {}

    def fake_train(*, max_iter: int):
        calls["train_max_iter"] = max_iter
        return SimpleNamespace(
            y_test=[0, 1],
            y_pred=[0, 1],
            y_proba=[0.2, 0.8],
            pipeline="fake-pipeline",
        )

    fake_metrics = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "pr_auc": 1.0,
    }

    def fake_log_run_to_mlflow(**kwargs):
        calls["mlflow_kwargs"] = kwargs

    monkeypatch.setattr(modeling_main, "parse_args", lambda: fake_args)
    monkeypatch.setattr(modeling_main, "train", fake_train)
    monkeypatch.setattr(modeling_main, "evaluate_predictions", lambda **_: fake_metrics)
    monkeypatch.setattr(modeling_main, "log_run_to_mlflow", fake_log_run_to_mlflow)

    modeling_main.main()

    output = capsys.readouterr().out
    assert calls["train_max_iter"] == 33
    assert calls["mlflow_kwargs"] == {
        "max_iter": 33,
        "metrics": fake_metrics,
        "pipeline": "fake-pipeline",
        "tracking_uri": "http://127.0.0.1:5000",
        "experiment_name": "credit-scoring-exp",
        "run_name": "run-1",
    }
    assert "MLflow logging completed." in output
