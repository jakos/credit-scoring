import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from credit_scoring.training import main as modeling_main


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

    captured = capsys.readouterr()
    output = captured.out
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert events[0]["event"] == "training_completed"
    assert events[0]["metrics"] == fake_metrics
    assert events[0]["sample_count"] == 4
    assert [event["event"] for event in events] == ["training_completed"]
    assert "http://" not in captured.err
    assert "settings" not in captured.err
    assert "y_test" not in captured.err
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

    captured = capsys.readouterr()
    output = captured.out
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert events[0]["event"] == "training_completed"
    assert events[0]["metrics"] == fake_metrics
    assert events[0]["sample_count"] == 2
    assert [event["event"] for event in events] == [
        "training_completed",
        "mlflow_logging_completed",
    ]
    assert "http://" not in captured.err
    assert "settings" not in captured.err
    assert "y_test" not in captured.err
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


def test_environment_fallback_and_flag_precedence(monkeypatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "71")
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_EXPERIMENT", "environment")
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = modeling_main.parse_args()
    assert args.max_iter == 71
    assert args.mlflow_tracking_uri == "http://localhost:5000"
    assert args.mlflow_experiment == "environment"
    assert not args.enable_mlflow
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--max-iter",
            "22",
            "--mlflow-tracking-uri",
            "http://localhost:6000",
            "--mlflow-experiment",
            "flag",
        ],
    )
    args = modeling_main.parse_args()
    assert args.max_iter == 22
    assert args.mlflow_tracking_uri == "http://localhost:6000"
    assert args.mlflow_experiment == "flag"


def test_cli_uses_environment_dataset_without_enabling_mlflow(
    tmp_path, monkeypatch, capsys
) -> None:
    from unittest.mock import Mock

    from credit_scoring.training.train import train as real_train

    csv = tmp_path / "dataset.csv"
    import pandas as pd

    pd.DataFrame(
        {"ID": range(40), "a": range(40), "default.payment.next.month": [0, 1] * 20}
    ).to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREDIT_SCORING_DATASET", "dataset.csv")
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", "http://private:5000")
    monkeypatch.setattr(sys, "argv", ["prog"])
    trainer = Mock(wraps=real_train)
    tracker = Mock()
    monkeypatch.setattr(modeling_main, "train", trainer)
    monkeypatch.setattr(modeling_main, "log_run_to_mlflow", tracker)
    modeling_main.main()
    from pathlib import Path

    trainer.assert_called_once_with(max_iter=1000, dataset_path=Path("dataset.csv"))
    tracker.assert_not_called()
    assert "Model training completed." in capsys.readouterr().out


def test_help_does_not_validate_train_or_contact_services(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from unittest.mock import Mock

    import pytest

    monkeypatch.setenv("LOG_LEVEL", "invalid")
    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    trainer = Mock()
    tracker = Mock()
    monkeypatch.setattr(modeling_main, "train", trainer)
    monkeypatch.setattr(modeling_main, "log_run_to_mlflow", tracker)
    with pytest.raises(SystemExit) as result:
        modeling_main.main()
    assert result.value.code == 0
    trainer.assert_not_called()
    tracker.assert_not_called()
    help_text = capsys.readouterr().out
    assert "--max-iter" in help_text
    assert "Registered model name for lifecycle promotion" in help_text
    assert "Path to the lifecycle promotion policy" in help_text


def test_explicit_iterations_override_invalid_environment(monkeypatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "invalid")
    monkeypatch.setattr(sys, "argv", ["prog", "--max-iter", "20"])
    assert modeling_main.parse_args().max_iter == 20


def test_lifecycle_settings_and_tracking_requirement(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr(sys, "argv", ["prog", "--lifecycle"])
    with pytest.raises(SystemExit):
        modeling_main.parse_args()
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("CREDIT_SCORING_REGISTRY_ENABLED", "true")
    monkeypatch.setenv("CREDIT_SCORING_REGISTERED_MODEL_NAME", "environment")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--registered-model-name", "flag", "--promotion-policy-path", "custom.toml"],
    )
    args = modeling_main.parse_args()
    assert args.lifecycle
    assert args.settings.registered_model_name == "flag"
    assert str(args.settings.promotion_policy_path) == "custom.toml"
    monkeypatch.setattr(sys, "argv", ["prog", "--no-lifecycle"])
    assert not modeling_main.parse_args().lifecycle


def test_cli_dispatches_shared_lifecycle(monkeypatch, capsys) -> None:
    from pathlib import Path
    from unittest.mock import Mock

    import credit_scoring.data.loader as loader
    import credit_scoring.registry.lifecycle as lifecycle
    import credit_scoring.registry.mlflow_adapter as adapter
    from credit_scoring.registry.promotion import Decision

    monkeypatch.setenv("CREDIT_SCORING_DATASET", "fixture.csv")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--lifecycle",
            "--mlflow-tracking-uri",
            "http://localhost:5000",
            "--registered-model-name",
            "risk",
            "--max-iter",
            "40",
        ],
    )
    frame = object()
    load = Mock(return_value=frame)
    factory = Mock()
    run = Mock(return_value=Decision(True, {}, ()))
    monkeypatch.setattr(loader, "load_credit_data", load)
    monkeypatch.setattr(adapter, "MLflowRegistry", factory)
    monkeypatch.setattr(lifecycle, "run_lifecycle", run)
    modeling_main.main()
    load.assert_called_once_with(Path("fixture.csv"))
    factory.assert_called_once_with("http://localhost:5000", "credit-scoring", "risk")
    assert run.call_args.args[:2] == (frame, factory.return_value)
    assert run.call_args.kwargs == {"max_iter": 40}
    assert "Promoted: True" in capsys.readouterr().out
