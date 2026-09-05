"""Deterministic tests for LinkedIn OAuth state and token handling."""

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from database.database import Base
from database.models import UserSettings
from services.linkedin_oauth_service import (
    LinkedInOAuthService,
    LinkedInToken,
    OAuthError,
)


def make_settings(**values: object) -> Settings:
    return Settings(
        linkedin_client_id="test-client",
        linkedin_client_secret="test-client-secret",
        linkedin_token_encryption_key=Fernet.generate_key().decode(),
        **values,
    )


def make_session() -> tuple[Session, object]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)(), engine


def test_oauth_initialization_and_authorization_url() -> None:
    service = LinkedInOAuthService(make_settings())

    url = service.get_authorization_url()
    query = parse_qs(urlparse(url).query)

    assert url.startswith(service.authorization_endpoint)
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["test-client"]
    assert query["scope"] == ["openid profile email w_member_social"]
    assert query["state"]


def test_state_is_one_time_and_rejects_missing_or_invalid_state() -> None:
    service = LinkedInOAuthService(make_settings())
    state = parse_qs(urlparse(service.get_authorization_url()).query)["state"][0]

    service.validate_state(state)
    with pytest.raises(OAuthError, match="Invalid or expired"):
        service.validate_state(state)
    with pytest.raises(OAuthError, match="Missing OAuth state"):
        service.validate_state(None)


def test_expired_state_is_rejected() -> None:
    now = datetime.now(timezone.utc)
    service = LinkedInOAuthService(make_settings(), clock=lambda: now)
    state = parse_qs(urlparse(service.get_authorization_url()).query)["state"][0]
    service._states[state] = now - timedelta(seconds=1)

    with pytest.raises(OAuthError, match="Invalid or expired"):
        service.validate_state(state)


def test_token_exchange_success_and_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/oauth/v2/accessToken")
        return httpx.Response(200, json={"access_token": "test-token", "expires_in": 3600, "scope": "openid"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = LinkedInOAuthService(make_settings(), client=client)
    state = parse_qs(urlparse(service.get_authorization_url()).query)["state"][0]

    token = service.exchange_code_for_token("test-code", state)

    assert isinstance(token, LinkedInToken)
    assert token.expires_in == 3600


def test_token_exchange_failure_and_missing_code() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(400)))
    service = LinkedInOAuthService(make_settings(), client=client)
    state = parse_qs(urlparse(service.get_authorization_url()).query)["state"][0]

    with pytest.raises(OAuthError, match="token exchange failed"):
        service.exchange_code_for_token("test-code", state)
    state = parse_qs(urlparse(service.get_authorization_url()).query)["state"][0]
    with pytest.raises(OAuthError, match="Missing authorization code"):
        service.exchange_code_for_token("", state)


def test_encrypted_token_persistence_status_expiry_and_disconnect() -> None:
    session, engine = make_session()
    settings = make_settings()
    service = LinkedInOAuthService(settings)
    token = LinkedInToken(access_token="test-token", expires_in=3600, scope="openid profile")

    service.store_token(session, token, "person-123")
    record = session.query(UserSettings).one()
    assert record.linkedin_encrypted_access_token != token.access_token
    assert service.get_access_token(session) == token.access_token
    status = service.get_connection_status(session)
    assert status.connected is True
    assert status.account_id == "person-123"
    assert "access_token" not in status.model_dump()

    service.disconnect(session)
    assert service.get_connection_status(session).connected is False
    with pytest.raises(OAuthError, match="not connected"):
        service.get_access_token(session)
    session.close()
    engine.dispose()


def test_expired_token_is_rejected_and_secrets_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    now = datetime.now(timezone.utc)
    session, engine = make_session()
    service = LinkedInOAuthService(make_settings(), clock=lambda: now)
    service.store_token(session, LinkedInToken(access_token="test-token", expires_in=1), "person")
    session.query(UserSettings).one().linkedin_expires_at = now - timedelta(seconds=1)
    session.commit()

    with pytest.raises(OAuthError, match="expired"):
        service.get_access_token(session)
    assert "test-token" not in caplog.text
    session.close()
    engine.dispose()