"""Structured logging setup using the standard library (no extra deps)."""

from __future__ import annotations

import json
import logging
import sys

from researchmind.config import get_settings


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record (machine-parseable logs)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Install a root handler according to settings; quiet noisy libraries."""
    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)

    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"
            )
        )

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())

    for name in ("httpx", "httpcore", "urllib3", "tavily"):
        logging.getLogger(name).setLevel(logging.WARNING)
