"""Structured application dependency health checks."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import Settings, get_settings


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class HealthReport(BaseModel):
    status: HealthStatus
    checks: dict[str, str] = Field(default_factory=dict)


class HealthService:
    """Check application dependencies without exposing configuration secrets."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def check_database(self, session: Session) -> str:
        """Return database health from a harmless connectivity query."""

        try:
            session.execute(text("SELECT 1"))
            return "healthy"
        except Exception:
            return "unhealthy"

    def check(self, session: Session | None = None, scheduler: object | None = None, linkedin_connected: bool = False) -> HealthReport:
        """Return aggregate HEALTHY, DEGRADED, or UNHEALTHY status."""

        checks = {
            "configuration": "healthy",
            "llm": "configured" if self.settings.llm_provider == "mock" or bool(self.settings.gemini_api_key or self.settings.llm_api_key) else "unconfigured",
            "scheduler": "enabled" if self.settings.scheduler_enabled else "disabled",
            "linkedin": "connected" if linkedin_connected else "not_connected",
        }
        if session is not None:
            checks["database"] = self.check_database(session)
        status = HealthStatus.HEALTHY
        if checks.get("database") == "unhealthy" or checks["llm"] == "unconfigured":
            status = HealthStatus.UNHEALTHY
        elif not linkedin_connected and not self.settings.linkedin_publishing_enabled:
            status = HealthStatus.HEALTHY
        elif not linkedin_connected:
            status = HealthStatus.DEGRADED
        return HealthReport(status=status, checks=checks)