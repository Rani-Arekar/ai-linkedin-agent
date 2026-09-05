"""Small, explicit CRUD operations used by application services."""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import AgentLog, Post, Source, Topic, UserSettings


def create_topic(session: Session, **values: object) -> Topic:
    """Create and commit a researched topic."""

    topic = Topic(**values)
    session.add(topic)
    session.commit()
    session.refresh(topic)
    return topic


def list_topics(session: Session, limit: int = 100) -> list[Topic]:
    """Return newest topics first."""

    statement = select(Topic).order_by(Topic.created_at.desc()).limit(limit)
    return list(session.scalars(statement))


def get_topic(session: Session, topic_id: int) -> Topic | None:
    """Find a topic by primary key."""

    return session.get(Topic, topic_id)


def create_post(session: Session, **values: object) -> Post:
    """Create and commit a generated post."""

    post = Post(**values)
    session.add(post)
    session.commit()
    session.refresh(post)
    return post


def update_post_status(
    session: Session,
    post_id: int,
    status: str,
    *,
    approved_at: datetime | None = None,
    published_at: datetime | None = None,
    linkedin_post_id: str | None = None,
) -> Post | None:
    """Update lifecycle fields on a post and return it."""

    post = session.get(Post, post_id)
    if post is None:
        return None
    post.status = status
    if approved_at is not None:
        post.approved_at = approved_at
    if published_at is not None:
        post.published_at = published_at
    if linkedin_post_id is not None:
        post.linkedin_post_id = linkedin_post_id
    session.commit()
    session.refresh(post)
    return post


def create_source(session: Session, **values: object) -> Source:
    """Create and commit a research source."""

    source = Source(**values)
    session.add(source)
    session.commit()
    session.refresh(source)
    return source


def create_agent_log(session: Session, **values: object) -> AgentLog:
    """Create and commit an agent execution log."""

    log = AgentLog(**values)
    session.add(log)
    session.commit()
    session.refresh(log)
    return log


def get_settings(session: Session) -> UserSettings:
    """Return the singleton settings row, creating defaults when needed."""

    settings = session.scalars(select(UserSettings).limit(1)).first()
    if settings is None:
        settings = UserSettings()
        session.add(settings)
        session.commit()
        session.refresh(settings)
    return settings


def update_settings(session: Session, **values: object) -> UserSettings:
    """Update the singleton settings row with validated application values."""

    settings = get_settings(session)
    for field, value in values.items():
        if not hasattr(settings, field):
            raise ValueError(f"Unknown user setting: {field}")
        setattr(settings, field, value)
    session.commit()
    session.refresh(settings)
    return settings


def get_linkedin_connection(session: Session) -> UserSettings:
    """Return the singleton settings row containing local connection metadata."""

    return get_settings(session)


def save_linkedin_connection(session: Session, **values: object) -> UserSettings:
    """Persist encrypted LinkedIn connection data in the existing settings row."""

    settings = get_settings(session)
    for field, value in values.items():
        if not field.startswith("linkedin_"):
            raise ValueError(f"Invalid LinkedIn connection field: {field}")
        setattr(settings, field, value)
    session.commit()
    session.refresh(settings)
    return settings


def disconnect_linkedin(session: Session) -> UserSettings:
    """Deactivate and remove local LinkedIn token/identity data."""

    settings = get_settings(session)
    settings.linkedin_encrypted_access_token = None
    settings.linkedin_encrypted_refresh_token = None
    settings.linkedin_account_id = None
    settings.linkedin_scopes = None
    settings.linkedin_expires_at = None
    settings.linkedin_connected_at = None
    settings.linkedin_active = False
    session.commit()
    session.refresh(settings)
    return settings


def get_linkedin_connection(session: Session) -> UserSettings:
    """Return the singleton settings row containing local connection metadata."""

    return get_settings(session)


def save_linkedin_connection(session: Session, **values: object) -> UserSettings:
    """Persist encrypted LinkedIn connection data in the existing settings row."""

    settings = get_settings(session)
    for field, value in values.items():
        if not field.startswith("linkedin_"):
            raise ValueError(f"Invalid LinkedIn connection field: {field}")
        setattr(settings, field, value)
    session.commit()
    session.refresh(settings)
    return settings


def disconnect_linkedin(session: Session) -> UserSettings:
    """Deactivate and remove local LinkedIn token/identity data."""

    settings = get_settings(session)
    settings.linkedin_encrypted_access_token = None
    settings.linkedin_encrypted_refresh_token = None
    settings.linkedin_account_id = None
    settings.linkedin_scopes = None
    settings.linkedin_expires_at = None
    settings.linkedin_connected_at = None
    settings.linkedin_active = False
    session.commit()
    session.refresh(settings)
    return settings