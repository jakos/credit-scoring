"""JSON application logging isolated from root and Airflow handlers."""

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Serialize common record fields and explicitly supplied event context."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _ApplicationHandler(logging.StreamHandler):
    """Identify the handler owned by this package across repeated setup."""


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure only the package namespace, preserving all external handlers."""
    logger = logging.getLogger("credit_scoring")
    handlers = [h for h in logger.handlers if isinstance(h, _ApplicationHandler)]
    if handlers:
        handler = handlers[0]
        # Entry points can run repeatedly with a different stderr capture/redirect.
        handler.stream = sys.stderr
    else:
        handler = _ApplicationHandler(sys.stderr)
        logger.addHandler(handler)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# Only fixed application messages may cross the diagnostic logging boundary.
# Third-party exceptions may interpolate entire input frames, URIs or credentials.
_SAFE_FAILURE_MESSAGES = frozenset(
    {
        "No champion version is available",
        "Required input feature metadata is missing",
        "Feature names must be nonempty strings",
        "Incompatible input feature metadata",
        "Model must support predict_proba",
        "Binary numeric class metadata is required",
        "Binary classes must include default label 1 exactly once",
        "Unsupported explanation contract",
        "Registry lookup or model loading failed",
        "Invalid probability shape",
        "Invalid probability values",
        "Missing explanation component",
        "Invalid explanation count",
        "Invalid explanation output",
    }
)


def failure_fields(exc: Exception) -> dict[str, str]:
    """Retain failure categories and trusted messages without serializing inputs."""
    cause = exc.__cause__ or exc
    message = str(exc) if str(exc) in _SAFE_FAILURE_MESSAGES else "Failure details redacted"
    cause_message = (
        str(cause) if str(cause) in _SAFE_FAILURE_MESSAGES else "Failure details redacted"
    )
    return {
        "exception_type": type(exc).__name__,
        "message": message,
        "cause_type": type(cause).__name__,
        "cause_message": cause_message,
    }
