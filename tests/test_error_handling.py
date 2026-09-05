"""Application-level safe error behavior tests."""

from services.linkedin_api_service import LinkedInAPIError
from services.linkedin_publish_service import PublishingError
from services.linkedin_oauth_service import OAuthError


def test_external_errors_have_safe_messages() -> None:
    messages = [
        str(LinkedInAPIError("LinkedIn request failed")),
        str(PublishingError("Publishing blocked")),
        str(OAuthError("OAuth state invalid")),
    ]

    assert all("token" not in message.lower() for message in messages)
    assert all("secret" not in message.lower() for message in messages)