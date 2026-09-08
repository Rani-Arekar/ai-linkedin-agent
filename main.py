"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import get_settings
from api.linkedin_oauth import router as linkedin_oauth_router
from database.database import SessionLocal, init_db
from database.models import UserSettings
from services.health_service import HealthService
from services.logging_service import configure_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize safe application resources and release them on shutdown."""

    configure_logging(settings.log_level)
    settings.validate_configuration()
    init_db()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(linkedin_oauth_router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Return basic service metadata."""

    return {"name": settings.app_name, "status": "ready"}


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Report whether the API process is running."""

    return {"status": "ok"}


@app.get("/health/live", tags=["system"])
def liveness() -> dict[str, str]:
    """Return a process-level liveness response."""

    return {"status": "ok"}


@app.get("/health/ready", tags=["system"])
def readiness() -> dict[str, object]:
    """Return dependency readiness without exposing secrets."""

    session = SessionLocal()

    try:
        user_settings = session.query(UserSettings).first()
        linkedin_connected = bool(
            user_settings and user_settings.linkedin_active
        )

        report = HealthService(settings).check(
            session=session,
            linkedin_connected=linkedin_connected,
        )

        return {
            "status": report.status.value,
            "checks": report.checks,
        }
    finally:
        session.close()