import json
import logging
from datetime import datetime

import pytest

from credit_scoring.logging import configure_logging


def test_json_fields_idempotence_and_external_handlers(capsys) -> None:
    root = logging.getLogger()
    root_handlers = root.handlers[:]
    airflow = logging.getLogger("airflow")
    airflow_handlers = airflow.handlers[:]
    logger = configure_logging("INFO")
    configure_logging("INFO")
    logging.getLogger("credit_scoring.training").info(
        "training_completed", extra={"fields": {"sample_count": 4, "metrics": {"accuracy": 0.75}}}
    )
    logger.debug("filtered")
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["severity"] == "INFO"
    assert record["logger"] == "credit_scoring.training"
    assert record["event"] == "training_completed"
    assert record["sample_count"] == 4
    assert record["metrics"] == {"accuracy": 0.75}
    assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None
    assert captured.out == ""
    assert root.handlers == root_handlers
    assert airflow.handlers == airflow_handlers


def test_exception_and_updated_severity(capsys) -> None:
    logger = configure_logging("ERROR")
    logger.warning("filtered")
    try:
        raise ValueError("example error")
    except ValueError:
        logger.exception("training_failed")
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert len(records) == 1
    assert records[0]["severity"] == "ERROR"
    assert records[0]["event"] == "training_failed"
    assert "Traceback" in records[0]["exception"]
    assert "ValueError: example error" in records[0]["exception"]
    configure_logging("DEBUG").debug("visible")
    assert json.loads(capsys.readouterr().err)["severity"] == "DEBUG"


def test_failure_fields_allow_only_trusted_messages(capsys: pytest.CaptureFixture[str]) -> None:
    from credit_scoring.logging import failure_fields

    cause = RuntimeError(
        "applicant-sentinel unknown-key-sentinel https://user:credential-sentinel@tracking-sentinel"
    )
    error = RuntimeError("Model prediction failed")
    error.__cause__ = cause
    logger = configure_logging()
    logger.error("serving_prediction_failed", extra={"fields": failure_fields(error)})
    serialized = capsys.readouterr().err
    assert "sentinel" not in serialized
    record = json.loads(serialized)
    assert record["exception_type"] == record["cause_type"] == "RuntimeError"
    assert record["cause_message"] == "Failure details redacted"
    assert "Traceback" not in serialized
    assert (
        failure_fields(ValueError("Invalid probability shape"))["cause_message"]
        == "Invalid probability shape"
    )
