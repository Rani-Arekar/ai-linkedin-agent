"""Pydantic schemas for validating database-facing API data."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    """Shared Pydantic configuration for ORM-backed response schemas."""

    model_config = ConfigDict(from_attributes=True)


class TopicCreate(BaseModel):
    """Input required to store a researched topic."""

    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=255)
    source_url: str = Field(min_length=1, max_length=2048)
    category: str = Field(min_length=1, max_length=100)
    importance_score: float | None = Field(default=None, ge=0, le=10)
    freshness_score: float | None = Field(default=None, ge=0, le=10)
    linkedin_score: float | None = Field(default=None, ge=0, le=10)


class TopicRead(SchemaBase, TopicCreate):
    """Serialized researched topic."""

    id: int
    created_at: datetime


class PostCreate(BaseModel):
    """Input required to store generated LinkedIn content."""

    topic_id: int = Field(gt=0)
    content: str = Field(min_length=1)
    status: str = Field(default="draft", min_length=1, max_length=50)
    quality_score: float | None = Field(default=None, ge=0, le=100)
    fact_check_status: str = Field(default="pending", min_length=1, max_length=50)
    similarity_score: float | None = Field(default=None, ge=0, le=1)


class PostRead(SchemaBase, PostCreate):
    """Serialized generated post."""

    id: int
    approved_at: datetime | None = None
    published_at: datetime | None = None
    linkedin_post_id: str | None = None
    created_at: datetime


class SourceCreate(BaseModel):
    """Input required to register a research source."""

    name: str = Field(min_length=1, max_length=255)
    url: str = Field(min_length=1, max_length=2048)
    source_type: str = Field(min_length=1, max_length=50)


class SourceRead(SchemaBase, SourceCreate):
    """Serialized research source."""

    id: int
    last_checked: datetime | None = None


class AgentLogCreate(BaseModel):
    """Input required to record an agent execution."""

    agent_name: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=255)
    status: str = Field(min_length=1, max_length=50)
    input_summary: str | None = None
    output_summary: str | None = None
    error_message: str | None = None
    execution_time: float | None = Field(default=None, ge=0)


class AgentLogRead(SchemaBase, AgentLogCreate):
    """Serialized agent execution log."""

    id: int
    created_at: datetime


class UserSettingsUpdate(BaseModel):
    """Validated user preference updates."""

    daily_publish_time: str | None = Field(default=None, pattern=r"^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    auto_publish: bool | None = None
    quality_threshold: int | None = Field(default=None, ge=0, le=100)
    similarity_threshold: float | None = Field(default=None, ge=0, le=1)
    preferred_categories: list[str] | None = None


class UserSettingsRead(SchemaBase):
    """Serialized user preferences."""

    id: int
    daily_publish_time: str
    auto_publish: bool
    quality_threshold: int
    similarity_threshold: float
    preferred_categories: list[str]


class LinkedInConnectionRead(BaseModel):
    """Safe LinkedIn connection metadata; token fields are intentionally absent."""

    connected: bool
    account_id: str | None = None
    connected_at: datetime | None = None
    expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)


class LinkedInConnectionRead(BaseModel):
    """Safe LinkedIn connection metadata; token fields are intentionally absent."""

    connected: bool
    account_id: str | None = None
    connected_at: datetime | None = None
    expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)