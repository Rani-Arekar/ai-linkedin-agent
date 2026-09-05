"""Explicitly gated publishing of human-approved LinkedIn drafts."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from config import Settings, get_settings
from database import crud
from database.models import Post
from services.linkedin_api_service import LinkedInAPIService
from services.linkedin_oauth_service import LinkedInOAuthService, OAuthError
from services.logging_service import log_event
import logging

logger = logging.getLogger(__name__)


class PublishingError(RuntimeError):
    """Raised when a draft fails a publishing safety gate."""


class LinkedInPublishService:
    """Publish only human-approved, AI-approved, connected, unposted drafts."""

    def __init__(
        self,
        api_service: LinkedInAPIService,
        oauth_service: LinkedInOAuthService,
        settings: Settings | None = None,
    ) -> None:
        self.api_service = api_service
        self.oauth_service = oauth_service
        self.settings = settings or get_settings()

    def publish_post(
        self,
        session: Session,
        post_id: int,
        human_approval: bool = False,
    ) -> Post:
        """Validate all gates, call LinkedIn, then persist confirmed publication."""

        post = session.get(Post, post_id)
        if post is None:
            log_event(logger, logging.WARNING, "Publishing blocked: draft missing", event="publish_blocked")
            raise PublishingError("Draft does not exist")
        if not self.settings.linkedin_publishing_enabled:
            log_event(logger, logging.INFO, "Publishing blocked: disabled", event="publish_blocked")
            raise PublishingError("LinkedIn publishing is disabled")
        if not human_approval or post.status != "APPROVED_FOR_PUBLISH":
            raise PublishingError("Human publishing approval is required")
        source_url = post.topic.source_url if post.topic is not None else None
        if not post.content.strip() or not source_url:
            raise PublishingError("Draft content and source are required")
        if post.fact_check_status.upper() not in {"PASS", "PASSED"}:
            raise PublishingError("Fact checking must pass before publishing")
        if post.quality_score is None or post.quality_score < self.settings.quality_threshold:
            raise PublishingError("Quality checking must pass before publishing")
        if post.linkedin_post_id:
            raise PublishingError("Draft has already been published")
        try:
            log_event(logger, logging.INFO, "Publishing started", event="publish_started")
            token = self.oauth_service.get_access_token(session)
            connection = self.oauth_service.get_connection_status(session)
            if not connection.account_id:
                raise PublishingError("LinkedIn account identity is unavailable")
            response = self.api_service.create_post(
                token,
                f"urn:li:person:{connection.account_id}",
                post.content,
                source_url,
            )
        except OAuthError as error:
            log_event(logger, logging.WARNING, "Publishing failed: OAuth unavailable", event="publish_failed")
            raise PublishingError(str(error)) from error
        except Exception:
            log_event(logger, logging.ERROR, "Publishing failed: provider error", event="publish_failed")
            raise
        updated = crud.update_post_status(
            session,
            post.id,
            "PUBLISHED",
            published_at=datetime.now(timezone.utc),
            linkedin_post_id=response.post_id,
        )
        if updated is None:
            raise PublishingError("Published draft could not be updated")
        log_event(logger, logging.INFO, "Publishing succeeded", event="publish_succeeded")
        return updated