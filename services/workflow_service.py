"""Thin execution facade for the compiled LangGraph workflow."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database import crud
from database.models import Topic
from graph.nodes import WorkflowDependencies
from graph.state import WorkflowState
from graph.workflow import build_workflow
from services.logging_service import log_event, run_id_context
from services.metrics_service import metrics


logger = logging.getLogger(__name__)


class WorkflowRunRecord(BaseModel):
    """Safe summary of a workflow execution for dashboard display."""

    run_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    retry_count: int = 0
    selected_topic: str | None = None
    error_summary: str | None = None


class WorkflowService:
    """Initialize isolated workflow state and invoke a compiled graph."""

    def __init__(
        self,
        dependencies: WorkflowDependencies,
        settings: Settings | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self.dependencies = dependencies
        self.settings = settings or get_settings()
        self.session_factory = session_factory
        self.workflow = build_workflow(dependencies)
        self._run_lock = Lock()
        self._run_history: list[WorkflowRunRecord] = []
        self._last_state: WorkflowState | None = None

    def run_workflow(self, run_id: str | None = None) -> WorkflowState:
        """Run one orchestration pass; no publishing service is available here."""

        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("A workflow run is already in progress")

        started_at = datetime.now(timezone.utc)
        selected_run_id = run_id or str(uuid4())
        context_token = run_id_context.set(selected_run_id)
        started_clock = perf_counter()

        log_event(
            logger,
            20,
            "Workflow started",
            event="started",
            run_id=selected_run_id,
        )

        initial: WorkflowState = {
            "run_id": selected_run_id,
            "retry_count": 0,
            "max_retries": self.settings.workflow_max_retries,
            "errors": [],
            "warnings": [],
            "agent_logs": [],
            "current_status": "STARTED",
        }

        try:
            state = self.workflow.invoke(initial)

            # Persist only successfully approved AI-generated drafts.
            if self.session_factory is not None:
                session = self.session_factory()
                try:
                    self._persist_generated_draft(session, state)
                finally:
                    session.close()

            self._last_state = state

            self._record_run(state, started_at)

            metrics.record_run(
                state.get("current_status", "FAILED"),
                perf_counter() - started_clock,
            )

            log_event(
                logger,
                20,
                "Workflow completed",
                event="completed",
                run_id=selected_run_id,
            )

            return state

        except Exception:
            metrics.record_run(
                "FAILED",
                perf_counter() - started_clock,
            )

            log_event(
                logger,
                40,
                "Workflow failed",
                event="failed",
                run_id=selected_run_id,
            )

            self._record_run(
                {
                    "run_id": selected_run_id,
                    "current_status": "FAILED",
                    "errors": ["Workflow execution failed"],
                },
                started_at,
            )

            raise

        finally:
            run_id_context.reset(context_token)
            self._run_lock.release()

    def start_workflow(self, run_id: str | None = None) -> WorkflowState:
        """Compatibility alias used by UI and scheduler callers."""

        return self.run_workflow(run_id)

    def get_workflow_status(self) -> dict[str, object]:
        """Return the latest safe workflow status and run identifier."""

        record = self._run_history[-1] if self._run_history else None

        return {
            "running": self._run_lock.locked(),
            "last_run": record,
            "last_state": self._last_state,
        }

    def get_last_run(self) -> WorkflowRunRecord | None:
        """Return the latest in-process run summary."""

        return self._run_history[-1] if self._run_history else None

    def get_run_history(
        self,
        limit: int = 20,
    ) -> list[WorkflowRunRecord]:
        """Return recent in-process run summaries, newest first."""

        return list(reversed(self._run_history[-limit:]))

    def _persist_generated_draft(
        self,
        session: Session,
        state: WorkflowState,
    ) -> None:
        """Persist an approved generated workflow result for human review."""

        # ---------------------------------------------------------
        # Only persist drafts when the complete AI workflow has
        # reached APPROVED status.
        # ---------------------------------------------------------
        if state.get("current_status") != "APPROVED":
            return

        selected_topic = state.get("selected_topic")
        generated_post = state.get("generated_post")

        if selected_topic is None or generated_post is None:
            return

        # ---------------------------------------------------------
        # Find the selected topic in the database.
        # Create it if it does not already exist.
        # ---------------------------------------------------------
        existing_topic = session.scalar(
            select(Topic).where(
                Topic.source_url == selected_topic.url
            )
        )

        if existing_topic is None:
            crud.create_topic(
                session,
                title=selected_topic.title,
                summary=selected_topic.summary,
                source=selected_topic.source,
                source_url=selected_topic.url,
                category=selected_topic.category,
                importance_score=selected_topic.importance_score,
                freshness_score=selected_topic.freshness_score,
                linkedin_score=selected_topic.linkedin_score,
            )

        # ---------------------------------------------------------
        # Extract evaluation results from workflow state.
        # ---------------------------------------------------------
        fact_check_result = state.get("fact_check_result")
        quality_check_result = state.get("quality_check_result")
        duplicate_check_result = state.get("duplicate_check_result")

        # ---------------------------------------------------------
        # Quality score
        # ---------------------------------------------------------
        quality_score = None

        if quality_check_result is not None:
            quality_score = getattr(
                quality_check_result,
                "overall_score",
                None,
            )

        # ---------------------------------------------------------
        # Fact-check status
        # ---------------------------------------------------------
        fact_check_status = "pending"

        if fact_check_result is not None:
            fact_check_status = getattr(
                fact_check_result,
                "overall_status",
                "pending",
            )

            if hasattr(fact_check_status, "value"):
                fact_check_status = fact_check_status.value

        # ---------------------------------------------------------
        # Duplicate/similarity score
        # ---------------------------------------------------------
        similarity_score = None

        if duplicate_check_result is not None:
            similarity_score = getattr(
                duplicate_check_result,
                "similarity_score",
                None,
            )

        # ---------------------------------------------------------
        # Save the generated post together with its AI evaluation
        # results.
        #
        # save_draft() keeps the database post as "draft".
        # We then mark it APPROVED below because the AI workflow
        # has successfully approved it.
        # ---------------------------------------------------------
        saved_post = self.dependencies.writer.save_draft(
            session,
            generated_post,
            selected_topic,
            quality_score=quality_score,
            fact_check_status=str(fact_check_status),
            similarity_score=similarity_score,
        )

        # ---------------------------------------------------------
        # Mark the post as AI-approved.
        #
        # IMPORTANT:
        # approved_at is NOT set here because this is still waiting
        # for the human approval step in the dashboard.
        # ---------------------------------------------------------
        crud.update_post_status(
            session,
            saved_post.id,
            "APPROVED",
        )

    def _record_run(
        self,
        state: WorkflowState,
        started_at: datetime,
    ) -> None:
        """Record a safe in-process workflow run summary."""

        selected = state.get("selected_topic")

        record = WorkflowRunRecord(
            run_id=state["run_id"],
            status=state.get("current_status", "FAILED"),
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            retry_count=state.get("retry_count", 0),
            selected_topic=getattr(
                selected,
                "title",
                None,
            ),
            error_summary="; ".join(
                state.get("errors", [])
            ) or None,
        )

        self._run_history.append(record)