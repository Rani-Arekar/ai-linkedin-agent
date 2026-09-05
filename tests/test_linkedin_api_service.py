"""Deterministic tests for the official LinkedIn API client."""

import httpx
import pytest

from config import Settings
from services.linkedin_api_service import LinkedInAPIError, LinkedInAPIService


def service(handler):
    return LinkedInAPIService(
        Settings(linkedin_api_version="202601"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_identity_request_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/userinfo"
        assert request.headers["X-Restli-Protocol-Version"] == "2.0.0"
        assert request.headers["LinkedIn-Version"] == "202601"
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(200, json={"sub": "person-123", "name": "Test Person"})

    user = service(handler).get_current_user("test-token")

    assert user.id == "person-123"
    assert user.name == "Test Person"


def test_post_request_uses_official_ugc_endpoint_and_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v2/ugcPosts"
        assert request.headers["X-Restli-Protocol-Version"] == "2.0.0"
        payload = request.read()
        assert b"lifecycleState" in payload
        assert b"urn:li:person:person-123" in payload
        return httpx.Response(201, headers={"X-RestLi-Id": "urn:li:share:123"})

    result = service(handler).create_post(
        "test-token", "urn:li:person:person-123", "A draft", "https://example.com/source"
    )

    assert result.post_id == "urn:li:share:123"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 429, 500, 503])
def test_http_errors_are_safe(status: int) -> None:
    result = service(lambda request: httpx.Response(status))

    with pytest.raises(LinkedInAPIError):
        result.get_current_user("test-token")


def test_timeout_connection_and_malformed_responses() -> None:
    timeout_service = service(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timeout")))
    with pytest.raises(LinkedInAPIError, match="timed out"):
        timeout_service.get_current_user("test-token")

    connection_service = service(lambda request: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    with pytest.raises(LinkedInAPIError, match="connection failed"):
        connection_service.get_current_user("test-token")

    malformed_service = service(lambda request: httpx.Response(200, text="not-json"))
    with pytest.raises(LinkedInAPIError, match="malformed"):
        malformed_service.get_current_user("test-token")


def test_missing_identity_and_post_id_are_rejected() -> None:
    missing_identity = service(lambda request: httpx.Response(200, json={}))
    with pytest.raises(LinkedInAPIError, match="identity"):
        missing_identity.get_current_user("test-token")

    missing_post_id = service(lambda request: httpx.Response(201))
    with pytest.raises(LinkedInAPIError, match="ID"):
        missing_post_id.create_post("test-token", "urn:li:person:person", "Text")


def test_invalid_input_and_token_are_rejected() -> None:
    client = service(lambda request: httpx.Response(200, json={"sub": "x"}))
    with pytest.raises(LinkedInAPIError, match="required"):
        client.get_current_user("")
    with pytest.raises(LinkedInAPIError, match="invalid"):
        client.create_post("token", "member:x", "Text")