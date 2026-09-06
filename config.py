"""Application configuration loaded from environment variables."""

from functools import lru_cache
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the LinkedIn content automation agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AI-Powered LinkedIn Content Automation Agent"
    app_env: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "mock"
    llm_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    llm_model: str = "gemini-2.0-flash"
    llm_timeout: float = Field(default=30.0, gt=0)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    linkedin_client_id: str | None = None
    linkedin_client_secret: str | None = None
    linkedin_redirect_uri: str = "http://localhost:8000/auth/linkedin/callback"

    database_url: str = "sqlite:///./linkedin_agent.db"

    daily_run_hour: int = Field(default=9, ge=0, le=23)
    daily_run_minute: int = Field(default=0, ge=0, le=59)
    quality_threshold: int = Field(default=80, ge=0, le=100)
    similarity_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    auto_publish: bool = False
    research_timeout_seconds: float = Field(default=15.0, gt=0)
    max_articles_per_source: int = Field(default=20, ge=1, le=100)
    research_keywords: str = (
        "artificial intelligence,AI,machine learning,deep learning,generative AI,"
        "large language model,LLM,computer vision,natural language processing,NLP,"
        "multimodal,AI agent,robotics,neural network,transformer,MLOps,responsible AI"
    )
    enabled_research_sources: str | None = None
    topic_selection_min_score: float = Field(default=4.0, ge=0, le=10)
    topic_selection_max_candidates: int = Field(default=20, ge=1, le=100)
    topic_selection_diversity_weight: float = Field(default=0.10, ge=0, le=1)
    topic_selection_freshness_weight: float = Field(default=0.15, ge=0, le=1)
    linkedin_post_max_characters: int = Field(default=3000, ge=100, le=10_000)
    linkedin_post_min_characters: int = Field(default=100, ge=1, le=10_000)
    linkedin_max_hashtags: int = Field(default=5, ge=1, le=10)
    workflow_max_retries: int = Field(default=2, ge=0, le=10)
    workflow_enable_duplicate_check: bool = True
    scheduler_enabled: bool = False
    schedule_time: str = "09:00"
    schedule_timezone: str = "Asia/Kolkata"
    max_concurrent_workflows: int = Field(default=1, ge=1, le=10)
    linkedin_scopes: str = "openid profile email w_member_social"
    linkedin_api_version: str = "202601"
    linkedin_token_encryption_key: str | None = None
    linkedin_publishing_enabled: bool = False
    linkedin_http_timeout: float = Field(default=20.0, gt=0)

    def safe_summary(self) -> dict[str, str]:
        """Return configuration metadata without secrets or token material."""

        return {
            "environment": self.app_env,
            "llm_provider": self.llm_provider,
            "database": self.database_url.split(":", 1)[0],
            "scheduler": "enabled" if self.scheduler_enabled else "disabled",
            "linkedin_publishing": "enabled" if self.linkedin_publishing_enabled else "disabled",
            "log_level": self.log_level,
        }

    def validate_configuration(self) -> None:
        """Validate runtime URLs and scheduler settings at explicit startup boundaries."""

        from datetime import datetime

        redirect = urlparse(self.linkedin_redirect_uri)
        if redirect.scheme not in {"http", "https"} or not redirect.netloc:
            raise ValueError("LINKEDIN_REDIRECT_URI must be an absolute HTTP(S) URL")
        try:
            datetime.strptime(self.schedule_time, "%H:%M")
        except ValueError as error:
            raise ValueError("SCHEDULE_TIME must use HH:MM format") from error
        try:
            ZoneInfo(self.schedule_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown schedule timezone: {self.schedule_timezone}") from error
        if self.linkedin_publishing_enabled and not all(
            (self.linkedin_client_id, self.linkedin_client_secret, self.linkedin_token_encryption_key)
        ):
            raise ValueError(
                "Publishing requires LinkedIn credentials and token encryption key"
            )



@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""

    return Settings()
