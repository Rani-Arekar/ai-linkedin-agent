"""Deterministic tests for explicitly gated LinkedIn publishing."""

from datetime import datetime, timezone

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from database.database import Base
from database.models import Post, Topic
from services.linkedin_api_service import LinkedInPostResponse
from services.linkedin_oauth_service import LinkedInConnectionStatus
from services.linkedin_publish_service import LinkedInPublishService, PublishingError


class OAuthStub:
    def __init__(self, connected: bool = True) -> None:
        self.connected = connected

    def get_access_token(self, session: Session) -> str:
        if not self.connected:
            raise RuntimeError("not connected")
        return "test-token"

    def get_connection_status(self, session: Session) -> LinkedInConnectionStatus:
        return LinkedInConnectionStatus(connected=self.connected, account_id="person-123")


class APIStub:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def create_post(self, *args: object, **kwargs: object) -> LinkedInPostResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("API failed")
        return LinkedInPostResponse(post_id="urn:li:share:123")


def make_session() -> tuple[Session, object]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    topic = Topic(
        title="AI topic",
        summary="Summary",
        source="Example",
        source_url="https://example.com/source",
        category="AI Research",
    )
    session.add(topic)
    session.flush()
    session.add(Post(topic_id=topic.id, content="Approved draft", status="APPROVED_FOR_PUBLISH", fact_check_status="PASS", quality_score=95))
    session.commit()
    return session, engine


def settings(enabled: bool = True) -> Settings:
    return Settings(
        linkedin_publishing_enabled=enabled,
        quality_threshold=80,
        linkedin_token_encryption_key=Fernet.generate_key().decode(),
    )


def publisher(api: APIStub, oauth: OAuthStub, enabled: bool = True) -> LinkedInPublishService:
    return LinkedInPublishService(api, oauth, settings(enabled))


def test_approved_human_approved_draft_publishes_and_updates_status() -> None:
    session, engine = make_session()
    api = APIStub()

    published = publisher(api, OAuthStub()).publish_post(session, 1, human_approval=True)

    assert published.status == "PUBLISHED"
    assert published.linkedin_post_id == "urn:li:share:123"
    assert published.published_at is not None
    assert api.calls == 1
    session.close()
    engine.dispose()


@pytest.mark.parametrize(
    "status,approval,message",
    [
        ("APPROVED", True, "Human publishing approval"),
        ("APPROVED_FOR_PUBLISH", False, "Human publishing approval"),
        ("DRAFT", True, "Human publishing approval"),
    ],
)
def test_human_approval_gate(status: str, approval: bool, message: str) -> None:
    session, engine = make_session()
    session.query(Post).one().status = status
    session.commit()
    api = APIStub()

    with pytest.raises(PublishingError, match=message):
        publisher(api, OAuthStub()).publish_post(session, 1, human_approval=approval)
    assert api.calls == 0
    session.close()
    engine.dispose()


def test_publishing_disabled_prevents_api_call() -> None:
    session, engine = make_session()
    api = APIStub()
    with pytest.raises(PublishingError, match="disabled"):
        publisher(api, OAuthStub(), enabled=False).publish_post(session, 1, human_approval=True)
    assert api.calls == 0
    session.close()
    engine.dispose()


def test_missing_connection_prevents_api_call() -> None:
    session, engine = make_session()
    api = APIStub()
    with pytest.raises((PublishingError, RuntimeError)):
        publisher(api, OAuthStub(False)).publish_post(session, 1, human_approval=True)
    assert api.calls == 0
    session.close()
    engine.dispose()


def test_fact_quality_duplicate_and_empty_gates() -> None:
    session, engine = make_session()
    api = APIStub()
    post = session.query(Post).one()
    post.fact_check_status = "FAIL"
    session.commit()
    with pytest.raises(PublishingError, match="Fact checking"):
        publisher(api, OAuthStub()).publish_post(session, 1, human_approval=True)
    post.fact_check_status = "PASS"
    post.quality_score = 10
    session.commit()
    with pytest.raises(PublishingError, match="Quality checking"):
        publisher(api, OAuthStub()).publish_post(session, 1, human_approval=True)
    post.quality_score = 95
    post.linkedin_post_id = "already-published"
    session.commit()
    with pytest.raises(PublishingError, match="already"):
        publisher(api, OAuthStub()).publish_post(session, 1, human_approval=True)
    assert api.calls == 0
    session.close()
    engine.dispose()


def test_api_failure_does_not_mark_post_published() -> None:
    session, engine = make_session()
    api = APIStub(fail=True)
    with pytest.raises(RuntimeError, match="API failed"):
        publisher(api, OAuthStub()).publish_post(session, 1, human_approval=True)
    assert session.query(Post).one().status == "APPROVED_FOR_PUBLISH"
    assert session.query(Post).one().linkedin_post_id is None
    session.close()
    engine.dispose()