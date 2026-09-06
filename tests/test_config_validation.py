"""Configuration validation and safe-summary tests."""

import pytest
from cryptography.fernet import Fernet

from config import Settings


def test_safe_configuration_summary_excludes_secrets() -> None:
    settings = Settings(
    llm_provider="mock",
    gemini_api_key="secret",
    linkedin_client_secret="client-secret",
)
    summary = settings.safe_summary()

    assert summary["llm_provider"] == "mock"
    assert "secret" not in str(summary)
    assert "client-secret" not in str(summary)


def test_valid_configuration_passes_explicit_validation() -> None:
    settings = Settings(
        linkedin_redirect_uri="https://example.com/callback",
        schedule_time="09:30",
        schedule_timezone="Asia/Kolkata",
    )

    settings.validate_configuration()


@pytest.mark.parametrize(
    "values, message",
    [
        ({"linkedin_redirect_uri": "not-a-url"}, "REDIRECT_URI"),
        ({"schedule_time": "bad"}, "HH:MM"),
        ({"schedule_timezone": "Not/AZone"}, "timezone"),
    ],
)
def test_invalid_runtime_configuration_is_rejected(values: dict[str, str], message: str) -> None:
    settings = Settings(**values)

    with pytest.raises(ValueError, match=message):
        settings.validate_configuration()


def test_enabled_publishing_requires_credentials_at_validation_boundary() -> None:
    settings = Settings(linkedin_publishing_enabled=True,
        linkedin_client_id=None,
        linkedin_client_secret=None,
        linkedin_token_encryption_key=None,)

    with pytest.raises(ValueError, match="credentials"):
        settings.validate_configuration()


def test_valid_publishing_configuration_passes() -> None:
    settings = Settings(
        linkedin_publishing_enabled=True,
        linkedin_client_id="client",
        linkedin_client_secret="secret",
        linkedin_token_encryption_key=Fernet.generate_key().decode(),
    )

    settings.validate_configuration()