import runpy
from importlib import import_module
from unittest.mock import Mock

import credit_scoring.modeling as legacy
from credit_scoring.data.loader import DEFAULT_DATA_PATH, load_credit_data
from credit_scoring.features.preprocess import TARGET_COLUMN, build_preprocessor
from credit_scoring.training import main


def test_legacy_exports() -> None:
    assert legacy.DEFAULT_DATA_PATH == DEFAULT_DATA_PATH
    assert legacy.load_credit_data is load_credit_data
    assert legacy.TARGET_COLUMN == TARGET_COLUMN
    assert legacy.build_preprocessor is build_preprocessor
    for old, new, names in [
        ("data_loader", "data.loader", ["load_credit_data", "DEFAULT_DATA_PATH"]),
        (
            "preprocess",
            "features.preprocess",
            ["split_features_target", "train_test_split_stratified", "build_preprocessor"],
        ),
        ("train", "training.train", ["train", "TrainingArtifacts"]),
        ("evaluate", "evaluation.metrics", ["evaluate_predictions"]),
        ("main", "training.main", ["main", "parse_args", "log_run_to_mlflow"]),
    ]:
        before = import_module(f"credit_scoring.modeling.{old}")
        after = import_module(f"credit_scoring.{new}")
        for name in names:
            assert getattr(before, name) is getattr(after, name)


def test_legacy_module_delegates(monkeypatch) -> None:
    delegate = Mock()
    monkeypatch.setattr(main, "main", delegate)
    # Remove an import performed by the compatibility test to emulate module execution.
    monkeypatch.delitem(__import__("sys").modules, "credit_scoring.modeling.main", raising=False)
    runpy.run_module("credit_scoring.modeling.main", run_name="__main__")
    delegate.assert_called_once_with()
