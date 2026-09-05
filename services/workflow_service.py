"""Thin execution facade for the compiled LangGraph workflow."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from time import perf_counter
from threading import Lock
from uuid import uuid4

from pydantic import BaseModel

from config import Settings, get_settings
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

    def __init__(self, dependencies: WorkflowDependencies, settings: Settings | None = None) -> None:
        self.dependencies = dependencies
        self.settings = settings or get_settings()
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
        log_event(logger, 20, "Workflow started", event="started", run_id=selected_run_id)
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
            self._last_state = state
            self._record_run(state, started_at)
            metrics.record_run(state.get("current_status", "FAILED"), perf_counter() - started_clock)
            log_event(logger, 20, "Workflow completed", event="completed", run_id=selected_run_id)
            return state
        except Exception:
            metrics.record_run("FAILED", perf_counter() - started_clock)
            log_event(logger, 40, "Workflow failed", event="failed", run_id=selected_run_id)
            self._record_run(
                {"run_id": selected_run_id, "current_status": "FAILED", "errors": ["Workflow execution failed"]},
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

    def get_run_history(self, limit: int = 20) -> list[WorkflowRunRecord]:
        """Return recent in-process run summaries, newest first."""

        return list(reversed(self._run_history[-limit:]))

    def _record_run(self, state: WorkflowState, started_at: datetime) -> None:
        selected = state.get("selected_topic")
        record = WorkflowRunRecord(
            run_id=state["run_id"],
            status=state.get("current_status", "FAILED"),
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            retry_count=state.get("retry_count", 0),
            selected_topic=getattr(selected, "title", None),
            error_summary="; ".join(state.get("errors", [])) or None,
        )
        self._run_history.append(record)