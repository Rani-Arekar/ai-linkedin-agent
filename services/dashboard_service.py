"""Read-only dashboard data facade over existing services and database tables."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import get_settings
from database import crud
from database.models import AgentLog, Post, Topic
from services.workflow_service import WorkflowService
from services.linkedin_api_service import LinkedInAPIService
from services.linkedin_oauth_service import LinkedInOAuthService
from services.linkedin_publish_service import LinkedInPublishService


class DashboardDataService:
    """Provide presentation-ready data without embedding business logic in the UI."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        workflow_service: WorkflowService | None = None,
        scheduler: Any | None = None,
        oauth_service: LinkedInOAuthService | None = None,
        api_service: LinkedInAPIService | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.workflow_service = workflow_service
        self.scheduler = scheduler
        self.oauth_service = oauth_service
        self.api_service = api_service

    def get_overview(self) -> dict[str, object]:
        """Return counts and latest workflow status from current data."""

        with self.session_factory() as session:
            topics = session.scalars(select(Topic)).all()
            posts = session.scalars(select(Post)).all()
            logs = session.scalars(select(AgentLog)).all()
        statuses = {status: sum(post.status.upper() == status for post in posts) for status in ("DRAFT", "APPROVED", "NEEDS_REVISION", "REJECTED", "FAILED")}
        workflow_status = self.workflow_service.get_workflow_status() if self.workflow_service else {"running": False, "last_run": None}
        scheduler_status = self.scheduler.get_scheduler_status() if self.scheduler else {"enabled": False, "running": False, "next_run": None}
        return {
            "workflow_status": workflow_status,
            "scheduler_status": scheduler_status,
            "research_articles": len(topics),
            "selected_topics": len(topics),
            "generated_drafts": len(posts),
            "agent_log_count": len(logs),
            **{key.lower(): value for key, value in statuses.items()},
        }

    def get_research_articles(self, category: str | None = None, source: str | None = None) -> list[Topic]:
        """Return persisted research topics with optional filters."""

        with self.session_factory() as session:
            statement = select(Topic).order_by(Topic.created_at.desc())
            if category:
                statement = statement.where(Topic.category == category)
            if source:
                statement = statement.where(Topic.source == source)
            return list(session.scalars(statement))

    def get_topics(self) -> list[Topic]:
        """Return selected-topic records from the existing topic table."""

        return self.get_research_articles()

    def get_drafts(self) -> list[Post]:
        """Return persisted generated drafts newest first."""

        with self.session_factory() as session:
            return list(session.scalars(select(Post).order_by(Post.created_at.desc())))

    def get_latest_evaluation(self) -> object | None:
        """Return the latest fact, quality, and final workflow evaluation."""

        if self.workflow_service is None:
            return None
        state = self.workflow_service.get_workflow_status().get("last_state")
        if not state:
            return None
        return {
            "fact_check": state.get("fact_check_result"),
            "quality_check": state.get("quality_check_result"),
            "duplicate_check": state.get("duplicate_check_result"),
            "final_status": state.get("current_status"),
        }

    def get_run_history(self, limit: int = 20) -> list[object]:
        """Return run summaries from the workflow facade."""

        return self.workflow_service.get_run_history(limit) if self.workflow_service else []

    def get_agent_logs(self, limit: int = 100) -> list[AgentLog]:
        """Return recent persisted agent logs without exposing secrets."""

        with self.session_factory() as session:
            statement = select(AgentLog).order_by(AgentLog.created_at.desc()).limit(limit)
            return list(session.scalars(statement))

    def run_workflow_now(self) -> object:
        """Delegate an explicit manual run to the existing workflow service."""

        if self.workflow_service is None:
            raise RuntimeError("Workflow service is not configured")
        return self.workflow_service.start_workflow()

    def get_linkedin_connection(self) -> object:
        """Return safe local LinkedIn connection status."""

        if self.oauth_service is None:
            return {"connected": False, "account_id": None, "scopes": []}
        with self.session_factory() as session:
            return self.oauth_service.get_connection_status(session)

    def get_linkedin_authorization_url(self) -> str:
        """Create an authorization URL through the OAuth service."""

        if self.oauth_service is None:
            raise RuntimeError("LinkedIn OAuth is not configured")
        return self.oauth_service.get_authorization_url()

    def disconnect_linkedin(self) -> None:
        """Remove local LinkedIn connection state."""

        if self.oauth_service is None:
            raise RuntimeError("LinkedIn OAuth is not configured")
        with self.session_factory() as session:
            self.oauth_service.disconnect(session)

    def approve_for_publishing(self, post_id: int) -> object:
        """Record explicit human approval without publishing the draft."""

        with self.session_factory() as session:
            post = session.get(Post, post_id)
            if post is None:
                raise ValueError("Draft does not exist")
            if post.status != "APPROVED":
                raise ValueError("AI evaluation must be APPROVED first")
            return crud.update_post_status(
                session, post_id, "APPROVED_FOR_PUBLISH", approved_at=datetime.now(timezone.utc)
            )

    def publish_approved_post(self, post_id: int) -> object:
        """Delegate publishing to the gated publishing service."""

        if self.oauth_service is None or self.api_service is None:
            raise RuntimeError("LinkedIn publishing is not configured")
        publisher = LinkedInPublishService(
            self.api_service, self.oauth_service, get_settings()
        )
        with self.session_factory() as session:
            return publisher.publish_post(session, post_id, human_approval=True)