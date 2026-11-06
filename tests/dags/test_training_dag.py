import json
import subprocess
from datetime import datetime
from unittest.mock import Mock

import pytest
from pydantic import ValidationError


def test_graph_and_import_without_settings_validation(load_dag, monkeypatch) -> None:
    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "invalid")
    module = load_dag("training")
    dag = module.dag
    assert dag.dag_id == "credit_scoring_training"
    assert dag.schedule == "@daily"
    assert dag.start_date == datetime(2024, 1, 1)
    assert dag.catchup is False
    assert dag.tags == ["credit-scoring", "training"]
    assert set(dag.tasks) == {"validate_dataset", "train_model"}
    assert dag.tasks["validate_dataset"].downstream_task_ids == {"train_model"}
    assert dag.tasks["train_model"].upstream_task_ids == {"validate_dataset"}
    assert dag.tasks["validate_dataset"].python_callable is module._validate_dataset
    assert dag.tasks["train_model"].python_callable is module._run_training


@pytest.mark.parametrize(("override", "expected"), [(None, 300), ("73", 73)])
def test_training_arguments_and_inherited_dataset(
    override: str | None, expected: int, load_dag, dataset, monkeypatch, capsys
) -> None:
    from credit_scoring.training.main import parse_args

    module = load_dag("training")
    if override is not None:
        monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", override)

    def child(command: list[str], *, check: bool) -> None:
        assert check is True
        assert command == [
            "python",
            "-m",
            "credit_scoring.training.main",
            "--max-iter",
            str(expected),
        ]
        monkeypatch.setattr(__import__("sys"), "argv", command[2:])
        args = parse_args()
        assert args.settings.dataset == dataset
        assert args.max_iter == expected

    runner = Mock(side_effect=child)
    monkeypatch.setattr(module.subprocess, "run", runner)
    module._validate_dataset()
    module._run_training()
    runner.assert_called_once()
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [e["event"] for e in events] == ["dataset_validated", "training_task_completed"]
    assert events[1]["max_iter"] == expected


def test_training_task_errors(load_dag, tmp_path, monkeypatch) -> None:
    module = load_dag("training")
    monkeypatch.setenv("CREDIT_SCORING_DATASET", str(tmp_path / "missing.csv"))
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        module._validate_dataset()
    runner = Mock(side_effect=subprocess.CalledProcessError(1, "python"))
    monkeypatch.setattr(module.subprocess, "run", runner)
    with pytest.raises(subprocess.CalledProcessError):
        module._run_training()
    monkeypatch.setenv("CREDIT_SCORING_MAX_ITER", "0")
    with pytest.raises(ValidationError):
        module._run_training()
    assert runner.call_count == 1


def test_default_dataset_is_repository_relative(load_dag, tmp_path, monkeypatch) -> None:
    module = load_dag("training")
    monkeypatch.chdir(tmp_path)
    module._validate_dataset()
    assert module.DEFAULT_DATA_PATH.is_absolute()
    assert module.DEFAULT_DATA_PATH.is_file()
