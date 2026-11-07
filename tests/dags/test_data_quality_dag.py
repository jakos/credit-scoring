import json
from datetime import datetime

import pandas as pd
import pytest
from pydantic import ValidationError


def test_graph(load_dag) -> None:
    module = load_dag("data_quality")
    dag = module.dag
    assert dag.dag_id == "credit_scoring_data_quality"
    assert dag.schedule == "@daily"
    assert dag.start_date == datetime(2024, 1, 1)
    assert dag.catchup is False
    assert dag.tags == ["credit-scoring", "data-quality"]
    assert set(dag.tasks) == {"validate_schema", "validate_missing_ratio"}
    assert dag.tasks["validate_schema"].downstream_task_ids == {"validate_missing_ratio"}
    assert dag.tasks["validate_missing_ratio"].upstream_task_ids == {"validate_schema"}
    assert dag.tasks["validate_schema"].python_callable is module._validate_schema
    assert dag.tasks["validate_missing_ratio"].python_callable is module._validate_missing_ratio


def test_success_events(load_dag, dataset, capsys) -> None:
    module = load_dag("data_quality")
    module._validate_schema()
    module._validate_missing_ratio()
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [e["event"] for e in events] == ["schema_validated", "missing_ratio_validated"]
    assert events[0]["sample_count"] == 2
    assert events[1]["missing_ratio"] == 0
    assert events[1]["threshold"] == 0.10
    assert "records" not in events[0]


def test_missing_file_and_schema(load_dag, dataset, monkeypatch, tmp_path) -> None:
    module = load_dag("data_quality")
    pd.DataFrame({"ID": [1]}).to_csv(dataset, index=False)
    with pytest.raises(
        ValueError, match=r"Missing required columns.*AGE.*LIMIT_BAL.*default.payment.next.month"
    ):
        module._validate_schema()
    monkeypatch.setenv("CREDIT_SCORING_DATASET", str(tmp_path / "missing.csv"))
    for task in [module._validate_schema, module._validate_missing_ratio]:
        with pytest.raises(FileNotFoundError):
            task()


def test_ratio_threshold_and_validation(load_dag, dataset, monkeypatch) -> None:
    module = load_dag("data_quality")
    df = pd.read_csv(dataset)
    df.loc[0, "AGE"] = float("nan")
    df.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match=r"Missing ratio 0.1250.*threshold 0.1000"):
        module._validate_missing_ratio()
    monkeypatch.setenv("CREDIT_SCORING_MAX_MISSING_RATIO", "0.125")
    module._validate_missing_ratio()
    for value in ["-1", "1.1", "invalid"]:
        monkeypatch.setenv("CREDIT_SCORING_MAX_MISSING_RATIO", value)
        with pytest.raises(ValidationError):
            module._validate_missing_ratio()
