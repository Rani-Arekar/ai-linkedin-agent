"""Deterministic tests for the explicit Phase 9 scheduler."""

from threading import Event, Thread

from apscheduler.schedulers.background import BackgroundScheduler

from config import Settings
from scheduler.scheduler import SchedulerService


def test_scheduler_is_disabled_by_default() -> None:
    service = SchedulerService(lambda: "ok", Settings())

    assert service.start_scheduler() is False
    assert service.get_scheduler_status()["running"] is False


def test_scheduler_starts_and_registers_daily_job() -> None:
    service = SchedulerService(
        lambda: "ok",
        Settings(scheduler_enabled=True, schedule_time="09:30", schedule_timezone="Asia/Kolkata"),
    )

    assert service.start_scheduler() is True
    job = service.scheduler.get_job("daily-workflow")
    assert job is not None
    assert job.trigger.fields[5].expressions[0].first == 9
    assert job.trigger.fields[6].expressions[0].first == 30
    service.stop_scheduler()


def test_scheduler_can_stop_explicitly() -> None:
    service = SchedulerService(lambda: "ok", Settings(scheduler_enabled=True))
    service.start_scheduler()

    service.stop_scheduler()

    assert service.get_scheduler_status()["running"] is False


def test_manual_trigger_calls_workflow_without_scheduler() -> None:
    calls: list[str] = []
    service = SchedulerService(lambda: calls.append("run") or "result", Settings())

    assert service.run_workflow_now() == "result"
    assert calls == ["run"]
    assert service.get_scheduler_status()["running"] is False


def test_overlapping_workflow_is_skipped() -> None:
    started = Event()
    release = Event()
    calls: list[str] = []

    def runner() -> str:
        calls.append("run")
        started.set()
        release.wait(timeout=2)
        return "done"

    service = SchedulerService(runner, Settings())
    thread = Thread(target=service.run_workflow_now)
    thread.start()
    started.wait(timeout=2)

    assert service.run_workflow_now() is None
    release.set()
    thread.join(timeout=2)
    assert calls == ["run"]


def test_workflow_failure_is_captured() -> None:
    service = SchedulerService(
        lambda: (_ for _ in ()).throw(RuntimeError("failure")), Settings()
    )

    assert service.run_workflow_now() is None
    assert service.get_scheduler_status()["last_error"] == "Workflow execution failed"


def test_invalid_time_and_timezone_are_rejected() -> None:
    invalid_time = SchedulerService(
        lambda: None, Settings(scheduler_enabled=True, schedule_time="bad")
    )
    invalid_zone = SchedulerService(
        lambda: None,
        Settings(scheduler_enabled=True, schedule_timezone="Not/AZone"),
    )

    try:
        invalid_time.start_scheduler()
        assert False, "invalid schedule time should fail"
    except ValueError as error:
        assert "HH:MM" in str(error)
    try:
        invalid_zone.start_scheduler()
        assert False, "invalid timezone should fail"
    except ValueError as error:
        assert "timezone" in str(error)