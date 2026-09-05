"""Workflow run monitoring tests."""

from services.workflow_service import WorkflowRunRecord


def test_workflow_run_record_is_safe_and_serializable() -> None:
    record = WorkflowRunRecord(run_id="run-1", status="APPROVED", started_at="2026-09-04T09:00:00Z")

    assert record.run_id == "run-1"
    assert record.model_dump()["status"] == "APPROVED"
    assert "token" not in str(record.model_dump()).lower()