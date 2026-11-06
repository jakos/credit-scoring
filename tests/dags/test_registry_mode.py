import subprocess
import sys
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "name,call", [("training", "_run_training"), ("retraining", "_run_retraining")]
)
def test_registry_mode_inherits_configuration_and_propagates_errors(
    name, call, load_dag, dataset, monkeypatch
) -> None:
    from credit_scoring.training.main import parse_args

    monkeypatch.setenv("CREDIT_SCORING_REGISTRY_ENABLED", "true")
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("CREDIT_SCORING_MLFLOW_EXPERIMENT", "registry-experiment")
    monkeypatch.setenv("CREDIT_SCORING_REGISTERED_MODEL_NAME", "risk")
    monkeypatch.setenv("CREDIT_SCORING_PROMOTION_POLICY_PATH", "custom.toml")
    module = load_dag(name)
    calls = []

    def child(command, *, check):
        assert check
        assert "--lifecycle" in command
        monkeypatch.setattr(sys, "argv", command[2:])
        args = parse_args()
        assert args.lifecycle
        assert args.settings.dataset == dataset
        assert args.settings.registered_model_name == "risk"
        assert str(args.settings.promotion_policy_path) == "custom.toml"
        assert args.mlflow_experiment == "registry-experiment"
        assert args.mlflow_tracking_uri == "http://localhost:5000"
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", child)
    validate, train = module.dag.tasks.values()
    assert train.task_id in validate.downstream_task_ids
    validate.python_callable()
    getattr(module, call)()
    assert len(calls) == 1
    monkeypatch.setattr(
        module.subprocess, "run", Mock(side_effect=subprocess.CalledProcessError(1, "python"))
    )
    with pytest.raises(subprocess.CalledProcessError):
        getattr(module, call)()
