"""Metrics service tests."""

from services.metrics_service import MetricsService


def test_workflow_statuses_and_durations_are_counted() -> None:
    service = MetricsService()
    service.record_run("APPROVED", 2.0)
    service.record_run("FAILED", 4.0)
    service.record_run("PUBLISHED", 1.0)
    service.record_stage("research", 0.5)
    service.record_stage("research", 1.5)

    snapshot = service.snapshot()

    assert snapshot["total_runs"] == 3
    assert snapshot["approved_runs"] == 1
    assert snapshot["failed_runs"] == 1
    assert snapshot["published_runs"] == 1
    assert snapshot["published_posts"] == 1
    assert snapshot["average_workflow_duration"] == 2.3333
    assert snapshot["average_research_duration"] == 1.0