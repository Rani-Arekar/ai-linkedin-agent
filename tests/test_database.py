"""Tests for the Phase 2 database layer."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database import crud
from database.database import Base, _engine_options
from database.models import AgentLog, Post, Source, Topic, UserSettings
from database.schemas import PostCreate, TopicCreate, UserSettingsUpdate


@pytest.fixture
def session() -> Session:
    """Provide a clean in-memory database session for each test."""

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as database_session:
        yield database_session
    engine.dispose()


def test_models_and_crud_round_trip(session: Session) -> None:
    topic = crud.create_topic(
        session,
        title="New open model",
        summary="An important model release.",
        source="AI Lab",
        source_url="https://example.com/model",
        category="Generative AI",
        importance_score=9.0,
        freshness_score=10.0,
        linkedin_score=8.5,
    )
    post = crud.create_post(
        session,
        topic_id=topic.id,
        content="A concise technical post.",
        quality_score=92.0,
    )
    source = crud.create_source(
        session,
        name="AI Lab RSS",
        url="https://example.com/rss",
        source_type="rss",
    )
    log = crud.create_agent_log(
        session,
        agent_name="research",
        action="collect sources",
        status="success",
        execution_time=0.42,
    )

    assert crud.get_topic(session, topic.id).posts[0].id == post.id
    assert crud.list_topics(session)[0].id == topic.id
    assert source.id is not None
    assert log.status == "success"


def test_post_status_and_settings(session: Session) -> None:
    topic = crud.create_topic(
        session,
        title="Topic",
        summary="Summary",
        source="Source",
        source_url="https://example.com/topic",
        category="AI Research",
    )
    post = crud.create_post(session, topic_id=topic.id, content="Post")
    approved_at = datetime.now(timezone.utc)

    updated = crud.update_post_status(
        session, post.id, "approved", approved_at=approved_at
    )
    settings = crud.update_settings(
        session,
        quality_threshold=85,
        auto_publish=True,
        preferred_categories=["AI Agents"],
    )

    assert updated is not None
    assert updated.status == "approved"
    assert settings.quality_threshold == 85
    assert settings.auto_publish is True
    assert settings.preferred_categories == ["AI Agents"]
    assert crud.get_settings(session).id == settings.id


def test_unknown_setting_is_rejected(session: Session) -> None:
    with pytest.raises(ValueError, match="Unknown user setting"):
        crud.update_settings(session, not_a_setting=True)


def test_all_requested_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == {
        "topics",
        "posts",
        "sources",
        "agent_logs",
        "user_settings",
    }


def test_pydantic_schemas_validate_database_data() -> None:
    topic = TopicCreate(
        title="Valid topic",
        summary="A useful summary.",
        source="AI Lab",
        source_url="https://example.com/topic",
        category="AI Research",
    )
    post = PostCreate(topic_id=1, content="A valid post")
    settings = UserSettingsUpdate(
        daily_publish_time="09:30", quality_threshold=85, similarity_threshold=0.75
    )

    assert topic.category == "AI Research"
    assert post.status == "draft"
    assert settings.daily_publish_time == "09:30"


def test_database_options_support_sqlite_and_postgresql() -> None:
    assert _engine_options("sqlite:///:memory:") == {
        "connect_args": {"check_same_thread": False}
    }
    assert _engine_options("postgresql+psycopg://user:password@localhost/db") == {}