"""SQLAlchemy models for research, content, and execution history."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.database import Base


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class Topic(Base):
    """A researched AI topic that can produce one or more posts."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    importance_score: Mapped[float | None] = mapped_column(Float)
    freshness_score: Mapped[float | None] = mapped_column(Float)
    linkedin_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    posts: Mapped[list["Post"]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )


class Post(Base):
    """Generated LinkedIn content and its approval/publishing state."""

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="draft", nullable=False, index=True)
    quality_score: Mapped[float | None] = mapped_column(Float)
    fact_check_status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )
    similarity_score: Mapped[float | None] = mapped_column(Float)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    linkedin_post_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    topic: Mapped[Topic] = relationship(back_populates="posts")


class Source(Base):
    """A configured research source and its latest check time."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    last_checked: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentLog(Base):
    """Execution record for an individual agent action."""

    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    input_summary: Mapped[str | None] = mapped_column(Text)
    output_summary: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    execution_time: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class UserSettings(Base):
    """Persisted workflow and dashboard preferences."""

    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    daily_publish_time: Mapped[str] = mapped_column(String(5), default="09:00", nullable=False)
    auto_publish: Mapped[bool] = mapped_column(default=False, nullable=False)
    quality_threshold: Mapped[int] = mapped_column(Integer, default=80, nullable=False)
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.80, nullable=False)
    preferred_categories: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    linkedin_provider: Mapped[str | None] = mapped_column(String(50))
    linkedin_account_id: Mapped[str | None] = mapped_column(String(255))
    linkedin_encrypted_access_token: Mapped[str | None] = mapped_column(Text)
    linkedin_encrypted_refresh_token: Mapped[str | None] = mapped_column(Text)
    linkedin_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    linkedin_scopes: Mapped[str | None] = mapped_column(Text)
    linkedin_connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    linkedin_active: Mapped[bool] = mapped_column(default=False, nullable=False)
