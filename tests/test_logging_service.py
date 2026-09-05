"""Tests for structured context and secret redaction."""

import logging

from services.logging_service import SecretRedactingFilter, log_event


def test_secret_redacting_filter_removes_values() -> None:
    record = logging.LogRecord("test", logging.INFO, "", 0, "access_token=secret-value", (), None)

    SecretRedactingFilter().filter(record)

    assert "secret-value" not in record.getMessage()
    assert "REDACTED" in record.getMessage()


def test_log_event_includes_safe_context(caplog) -> None:
    logger = logging.getLogger("phase11-test")
    with caplog.at_level(logging.INFO):
        log_event(logger, logging.INFO, "workflow started", agent="research", event="started", run_id="run-123")

    assert "workflow started" in caplog.text
    assert "run-123" in caplog.text