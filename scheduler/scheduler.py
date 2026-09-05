"""Explicit APScheduler lifecycle for daily workflow execution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from threading import Lock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class SchedulerService:
    """Schedule a single daily workflow callback with overlap protection."""

    def __init__(
        self,
        workflow_runner: Callable[[], object],
        settings: Settings | None = None,
        scheduler: BackgroundScheduler | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.workflow_runner = workflow_runner
        self.scheduler = scheduler or BackgroundScheduler()
        self._run_lock = Lock()
        self._last_error: str | None = None

    def start_scheduler(self) -> bool:
        """Start the daily job explicitly when enabled; return whether it started."""

        if not self.settings.scheduler_enabled:
            logger.info("Scheduler disabled")
            return False
        self.settings.validate_configuration()
        if self.scheduler.running:
            return True
        hour, minute = self._schedule_parts()
        timezone = self._timezone()
        self.scheduler.add_job(
            self._run_scheduled_workflow,
            trigger=CronTrigger(hour=hour, minute=minute, timezone=timezone),
            id="daily-workflow",
            replace_existing=True,
            max_instances=self.settings.max_concurrent_workflows,
            coalesce=True,
            misfire_grace_time=3600,
        )
        self.scheduler.start()
        return True

    def stop_scheduler(self) -> None:
        """Stop the scheduler without waiting for a running workflow."""

        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def run_workflow_now(self) -> object | None:
        """Run the workflow manually without changing the daily schedule."""

        return self._run_workflow("manual")

    def get_scheduler_status(self) -> dict[str, object]:
        """Return safe scheduler state suitable for a dashboard."""

        job = self.scheduler.get_job("daily-workflow")
        return {
            "enabled": self.settings.scheduler_enabled,
            "running": self.scheduler.running,
            "next_run": job.next_run_time if job else None,
            "schedule_time": self.settings.schedule_time,
            "timezone": self.settings.schedule_timezone,
            "workflow_running": self._run_lock.locked(),
            "last_error": self._last_error,
        }

    def _run_scheduled_workflow(self) -> object | None:
        return self._run_workflow("scheduled")

    def _run_workflow(self, trigger: str) -> object | None:
        if not self._run_lock.acquire(blocking=False):
            logger.info("Workflow already running; %s execution skipped.", trigger)
            return None
        try:
            self._last_error = None
            return self.workflow_runner()
        except Exception as error:
            self._last_error = "Workflow execution failed"
            logger.warning("Workflow execution failed: %s", type(error).__name__)
            return None
        finally:
            self._run_lock.release()

    def _schedule_parts(self) -> tuple[int, int]:
        try:
            parsed = datetime.strptime(self.settings.schedule_time, "%H:%M")
        except ValueError as error:
            raise ValueError("SCHEDULE_TIME must use HH:MM format") from error
        return parsed.hour, parsed.minute

    def _timezone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.settings.schedule_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown schedule timezone: {self.settings.schedule_timezone}") from error