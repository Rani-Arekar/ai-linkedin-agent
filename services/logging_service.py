"""Centralized redacting application logging helpers."""

from __future__ import annotations

import logging
import re
import sys
from contextvars import ContextVar

run_id_context: ContextVar[str] = ContextVar("run_id", default="-")
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|client[_-]?secret|access[_-]?token|refresh[_-]?token|password|authorization)\s*[:=]\s*[^\s,;]+"
)


class SecretRedactingFilter(logging.Filter):
    """Remove credential-like values from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET_PATTERN.sub(r"\1=[REDACTED]", str(record.msg))
        record.args = ()
        return True


class ContextFormatter(logging.Formatter):
    """Format logs with stable run context and structured event fields."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


def configure_logging(level: str = "INFO") -> None:
    """Configure one redacting stdout handler without duplicating handlers."""

    root = logging.getLogger()
    root.setLevel(level.upper())
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ContextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handler.addFilter(SecretRedactingFilter())
        root.addHandler(handler)


def log_event(logger: logging.Logger, level: int, message: str, *, agent: str = "-", event: str = "-", run_id: str | None = None) -> None:
    """Emit a structured event with safe context."""

    context = f"run_id={run_id or run_id_context.get()} agent={agent} event={event}"
    logger.log(level, f"{context} {message}", extra={"agent": agent, "event": event, "run_id": run_id or run_id_context.get()})