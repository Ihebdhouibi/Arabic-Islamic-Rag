"""Central logging configuration.

Call ``configure_logging()`` once at process start, or rely on ``get_logger()`` which configures
lazily on first use. Level and format are controlled by ``SHAMELA_LOG_LEVEL`` (default ``INFO``)
and ``SHAMELA_LOG_JSON`` (default plain text).
"""

from __future__ import annotations

import json
import logging
import os
import sys

_TRUE = {"1", "true", "yes", "on"}
_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str] = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, json_format: bool | None = None) -> None:
    global _configured
    level = (level or os.getenv("SHAMELA_LOG_LEVEL") or "INFO").upper()
    if json_format is None:
        json_format = os.getenv("SHAMELA_LOG_JSON", "false").lower() in _TRUE

    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
