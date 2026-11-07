"""Minimal Airflow graph stubs; no scheduler, service or container is needed."""

import importlib.util
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def load_dag(monkeypatch) -> Callable[[str], ModuleType]:
    class DAG:
        current = None

        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            self.tasks = {}

        def __enter__(self) -> DAG:
            DAG.current = self
            return self

        def __exit__(self, *args: object) -> None:
            DAG.current = None

    class PythonOperator:
        def __init__(self, task_id: str, python_callable: Callable[[], None]) -> None:
            self.task_id = task_id
            self.python_callable = python_callable
            self.downstream_task_ids = set()
            self.upstream_task_ids = set()
            DAG.current.tasks[task_id] = self

        def __rshift__(self, other: PythonOperator) -> PythonOperator:
            self.downstream_task_ids.add(other.task_id)
            other.upstream_task_ids.add(self.task_id)
            return other

    for name, attributes in {
        "airflow": {"DAG": DAG},
        "airflow.operators": {},
        "airflow.operators.python": {"PythonOperator": PythonOperator},
        "pendulum": {"datetime": datetime},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(__import__("sys").modules, name, module)

    def load(name: str) -> ModuleType:
        path = Path(__file__).resolve().parents[2] / "dags" / f"credit_scoring_{name}_dag.py"
        spec = importlib.util.spec_from_file_location(f"test_dag_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def dataset(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "ID": [1, 2],
            "LIMIT_BAL": [100, 200],
            "AGE": [20, 30],
            "default.payment.next.month": [0, 1],
        }
    ).to_csv(path, index=False)
    monkeypatch.setenv("CREDIT_SCORING_DATASET", str(path))
    return path
