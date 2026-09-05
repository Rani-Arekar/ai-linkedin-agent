"""Secure LinkedIn OAuth 2.0 authorization-code flow and local token storage."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from threading import Lock
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database import crud


class OAuthError(RuntimeError):
    """Raised for safe OAuth configuration, state, or provider failures."""


class LinkedInToken(BaseModel):
    """Validated token response; never used as a dashboard response."""

    access_token: str = Field(min_length=1)
    expires_in: int = Field(gt=0)
    refresh_token: str | None = None
    refresh_token_expires_in: int | None = Field(default=None, gt=0)
    scope: str = ""


class LinkedInConnectionStatus(BaseModel):
    """Safe connection status without secret token values."""

    connected: bool
    account_id: str | None = None
    connected_at: datetime | None = None
    expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)


class LinkedInOAuthService:
    """Implement state-protected LinkedIn OAuth with encrypted local tokens."""

    authorization_endpoint = "https://www.linkedin.com/oauth/v2/authorization"
    token_endpoint = "https://www.linkedin.com/oauth/v2/accessToken"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or httpx.Client(timeout=self.settings.linkedin_http_timeout)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._states: dict[str, datetime] = {}
        self._state_lock = Lock()

    def get_authorization_url(self) -> str:
        """Create a LinkedIn authorization URL with a one-time CSRF state."""

        if not self.settings.linkedin_client_id:
            raise OAuthError("LinkedIn client ID is not configured")
        state = secrets.token_urlsafe(32)
        with self._state_lock:
            self._states[state] = self._clock() + timedelta(minutes=10)
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.settings.linkedin_client_id,
                "redirect_uri": self.settings.linkedin_redirect_uri,
                "state": state,
                "scope": self.settings.linkedin_scopes,
            }
        )
        return f"{self.authorization_endpoint}?{query}"

    def validate_state(self, state: str | None) -> None:
        """Validate and consume a non-expired OAuth state exactly once."""

        if not state:
            raise OAuthError("Missing OAuth state")
        with self._state_lock:
            expires_at = self._states.pop(state, None)
        if expires_at is None or expires_at <= self._clock():
            raise OAuthError("Invalid or expired OAuth state")

    def exchange_code_for_token(self, code: str, state: str | None = None) -> LinkedInToken:
        """Validate state and exchange an authorization code without logging secrets."""

        self.validate_state(state)
        if not code:
            raise OAuthError("Missing authorization code")
        if not self.settings.linkedin_client_id or not self.settings.linkedin_client_secret:
            raise OAuthError("LinkedIn OAuth credentials are not configured")
        try:
            response = self.client.post(
                self.token_endpoint,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self.settings.linkedin_client_id,
                    "client_secret": self.settings.linkedin_client_secret,
                    "redirect_uri": self.settings.linkedin_redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return LinkedInToken.model_validate(response.json())
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise OAuthError("LinkedIn token exchange failed") from error

    def store_token(
        self,
        session: Session,
        token: LinkedInToken,
        account_id: str,
        scopes: str | None = None,
    ) -> None:
        """Encrypt and store token material in the existing settings row."""

        cipher = self._cipher()
        now = self._clock()
        crud.save_linkedin_connection(
            session,
            linkedin_provider="linkedin",
            linkedin_account_id=account_id,
            linkedin_encrypted_access_token=cipher.encrypt(token.access_token.encode()).decode(),
            linkedin_encrypted_refresh_token=(
                cipher.encrypt(token.refresh_token.encode()).decode()
                if token.refresh_token
                else None
            ),
            linkedin_expires_at=now + timedelta(seconds=token.expires_in),
            linkedin_scopes=scopes or token.scope,
            linkedin_connected_at=now,
            linkedin_active=True,
        )

    def get_access_token(self, session: Session) -> str:
        """Decrypt the active access token for the API service only."""

        connection = crud.get_linkedin_connection(session)
        if not connection.linkedin_active or not connection.linkedin_encrypted_access_token:
            raise OAuthError("LinkedIn account is not connected")
        expires_at = as_utc(connection.linkedin_expires_at)
        if expires_at and expires_at <= as_utc(self._clock()):
            raise OAuthError("LinkedIn access token has expired")
        try:
            return self._cipher().decrypt(
                connection.linkedin_encrypted_access_token.encode()
            ).decode()
        except InvalidToken as error:
            raise OAuthError("Stored LinkedIn token is invalid") from error

    def get_connection_status(self, session: Session) -> LinkedInConnectionStatus:
        """Return safe connection metadata without decrypting or exposing tokens."""

        connection = crud.get_linkedin_connection(session)
        return LinkedInConnectionStatus(
            connected=bool(
                connection.linkedin_active
                and connection.linkedin_encrypted_access_token
            ),
            account_id=connection.linkedin_account_id,
            connected_at=as_utc(connection.linkedin_connected_at),
            expires_at=as_utc(connection.linkedin_expires_at),
            scopes=(connection.linkedin_scopes or "").split(),
        )

    def disconnect(self, session: Session) -> None:
        """Remove local connection material; provider-side revocation is not claimed."""

        crud.disconnect_linkedin(session)

    def _cipher(self) -> Fernet:
        if not self.settings.linkedin_token_encryption_key:
            raise OAuthError("LINKEDIN_TOKEN_ENCRYPTION_KEY is required for token storage")
        try:
            return Fernet(self.settings.linkedin_token_encryption_key.encode())
        except (ValueError, TypeError) as error:
            raise OAuthError("Invalid LinkedIn token encryption key") from error


def as_utc(value: datetime | None) -> datetime | None:
    """Normalize database timestamps for consistent aware comparisons."""

    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)