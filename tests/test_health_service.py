"""Health service tests using isolated SQLite sessions."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config import Settings
from services.health_service import HealthService, HealthStatus


def test_healthy_database_and_optional_linkedin() -> None:
    engine = create_engine("sqlite:///:memory:")
    with Session(engine) as session:
        report = HealthService(Settings()).check(session=session, linkedin_connected=False)

    assert report.status == HealthStatus.HEALTHY
    assert report.checks["database"] == "healthy"
    assert report.checks["linkedin"] == "not_connected"
    engine.dispose()


def test_missing_linkedin_is_degraded_when_publishing_enabled() -> None:
    settings = Settings(linkedin_publishing_enabled=True)
    report = HealthService(settings).check(linkedin_connected=False)

    assert report.status == HealthStatus.DEGRADED


def test_database_failure_is_unhealthy() -> None:
    class BrokenSession:
        def execute(self, statement):
            raise RuntimeError("database offline")

    report = HealthService(Settings()).check(session=BrokenSession())

    assert report.status == HealthStatus.UNHEALTHY
    assert report.checks["database"] == "unhealthy"