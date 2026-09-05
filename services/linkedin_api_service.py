"""Official LinkedIn API client for identity and member post creation."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class LinkedInAPIError(RuntimeError):
    """Safe application-level LinkedIn API error."""


class LinkedInUser(BaseModel):
    """Authenticated member identity used to construct the author URN."""

    id: str = Field(min_length=1)
    name: str | None = None


class LinkedInPostResponse(BaseModel):
    """Normalized successful LinkedIn post response."""

    post_id: str = Field(min_length=1)


class LinkedInAPIService:
    """Call documented LinkedIn endpoints using a supplied bearer token."""

    api_base = "https://api.linkedin.com"

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or httpx.Client(timeout=self.settings.linkedin_http_timeout)

    def get_current_user(self, access_token: str) -> LinkedInUser:
        """Retrieve identity through LinkedIn's OpenID Connect userinfo endpoint."""

        data = self._request("GET", "/v2/userinfo", access_token)
        try:
            return LinkedInUser(id=str(data["sub"]), name=data.get("name"))
        except (KeyError, TypeError, ValueError) as error:
            raise LinkedInAPIError("LinkedIn identity response was malformed") from error

    def create_post(
        self,
        access_token: str,
        author_urn: str,
        content: str,
        source_url: str | None = None,
    ) -> LinkedInPostResponse:
        """Create a text or article share through the official UGC Posts API."""

        if not content.strip() or not author_urn.startswith("urn:li:person:"):
            raise LinkedInAPIError("LinkedIn post author or content is invalid")
        share_content: dict[str, Any] = {
            "shareCommentary": {"text": content},
            "shareMediaCategory": "NONE" if not source_url else "ARTICLE",
        }
        if source_url:
            share_content["media"] = [{"status": "READY", "originalUrl": source_url}]
        payload = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {"com.linkedin.ugc.ShareContent": share_content},
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }
        data = self._request("POST", "/v2/ugcPosts", access_token, json=payload, expect_json=False)
        post_id = data.get("post_id")
        if not post_id:
            raise LinkedInAPIError("LinkedIn post response did not include an ID")
        return LinkedInPostResponse(post_id=post_id)

    def _request(
        self,
        method: str,
        path: str,
        access_token: str,
        *,
        json: dict[str, Any] | None = None,
        expect_json: bool = True,
    ) -> dict[str, Any]:
        if not access_token:
            raise LinkedInAPIError("LinkedIn access token is required")
        try:
            response = self.client.request(
                method,
                f"{self.api_base}{path}",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                    "LinkedIn-Version": self.settings.linkedin_api_version,
                    "Content-Type": "application/json",
                },
                json=json,
            )
        except httpx.TimeoutException as error:
            raise LinkedInAPIError("LinkedIn request timed out") from error
        except httpx.HTTPError as error:
            raise LinkedInAPIError("LinkedIn connection failed") from error
        if response.status_code >= 400:
            if response.status_code == 429:
                raise LinkedInAPIError("LinkedIn rate limit reached")
            if response.status_code in {401, 403}:
                raise LinkedInAPIError("LinkedIn authorization was rejected")
            raise LinkedInAPIError(f"LinkedIn API request failed ({response.status_code})")
        if method == "POST" and not expect_json:
            post_id = response.headers.get("X-RestLi-Id")
            if not post_id:
                try:
                    post_id = response.json().get("id")
                except (ValueError, TypeError):
                    post_id = None
            return {"post_id": post_id} if post_id else {}
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            return payload
        except (ValueError, TypeError) as error:
            raise LinkedInAPIError("LinkedIn response was malformed") from error