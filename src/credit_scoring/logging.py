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
