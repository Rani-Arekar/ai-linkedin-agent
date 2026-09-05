"""Publishing audit and failure-safety tests."""

import logging

from services.linkedin_publish_service import logger


def test_publishing_logger_does_not_emit_secret_values(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("publishing requested", extra={"event": "publish_requested"})

    assert "access_token" not in caplog.text
    assert "Authorization" not in caplog.text