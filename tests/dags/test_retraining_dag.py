import json
import subprocess
from datetime import datetime
from unittest.mock import Mock

import pytest
from pydantic import ValidationError


def test_graph(load_dag) -> None:
    module = load_dag("retraining")
    dag = module.dag
    assert dag.dag_id == "credit_scoring_retraining"
    assert dag.schedule == "@weekly"
    assert dag.start_date == datetime(2024, 1, 1)
    assert dag.catchup is False
    assert dag.tags == ["credit-scoring", "retraining"]
    assert set(dag.tasks) == {"validate_dataset", "retrain_model"}
    assert dag.tasks["validate_dataset"].downstream_task_ids == {"retrain_model"}
    assert dag.tasks["retrain_model"].upstream_task_ids == {"validate_dataset"}
    assert dag.tasks["retrain_model"].python_callable is module._run_retraining


@pytest.mark.parametrize(
    ("override", "uri"),
    [(None, None), ("83", None), (None, "http://private:5000"), ("83", "http://private:5000")],
)
def test_retraining_arguments(
    override: str | None, uri: str | None, load_dag, dataset, monkeypatch, capsys
) -> None:
    from credit_scoring.training.main import parse_args

    module = load_dag("retraining")
    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "12")
    if override:
        monkeypatch.setenv("CREDIT_SCORING_RETRAIN_MAX_ITER", override)
    if uri:
        monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", uri)
        monkeypatch.setenv("CREDIT_SCORING_MLFLOW_EXPERIMENT", "experiment")
    expected = ["python", "-m", "credit_scoring.training.main", "--max-iter", override or "500"]
    if uri:
        expected += [
            "--enable-mlflow",
            "--mlflow-tracking-uri",
            uri,
            "--mlflow-experiment",
            "experiment",
        ]

    def child(command: list[str], *, check: bool) -> None:
        assert check is True
        monkeypatch.setattr(__import__("sys"), "argv", command[2:])
        args = parse_args()
        assert args.settings.dataset == dataset
        assert args.enable_mlflow is bool(uri)

    runner = Mock(side_effect=child)
    monkeypatch.setattr(module.subprocess, "run", runner)
    module._validate_dataset()
    module._run_retraining()
    runner.assert_called_once_with(expected, check=True)
    captured = capsys.readouterr().err
    assert "http://" not in captured
    events = [json.loads(line) for line in captured.splitlines()]
    assert [e["event"] for e in events] == ["dataset_validated", "retraining_task_completed"]
    assert events[1]["max_iter"] == int(override or "500")


def test_standard_mlflow_uri_does_not_enable_tracking(load_dag, monkeypatch) -> None:
    module = load_dag("retraining")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://shared:5000")
    runner = Mock()
    monkeypatch.setattr(module.subprocess, "run", runner)

    module._run_retraining()

    runner.assert_called_once_with(
        ["python", "-m", "credit_scoring.training.main", "--max-iter", "500"],
        check=True,
    )


def test_retraining_failures(load_dag, tmp_path, monkeypatch) -> None:
    module = load_dag("retraining")
    monkeypatch.setenv("CREDIT_SCORING_DATASET", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        module._validate_dataset()
    runner = Mock(side_effect=subprocess.CalledProcessError(1, "python"))
    monkeypatch.setattr(module.subprocess, "run", runner)
    with pytest.raises(subprocess.CalledProcessError):
        module._run_retraining()
    monkeypatch.setenv("CREDIT_SCORING_RETRAIN_MAX_ITER", "-1")
    with pytest.raises(ValidationError):
        module._run_retraining()
    assert runner.call_count == 1
