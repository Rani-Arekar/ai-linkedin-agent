"""Shared, serializable state for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    """Data passed between orchestration nodes; no service instances or secrets."""

    run_id: str
    research_articles: list[Any]
    selected_topic: Any
    article_analysis: Any
    generated_post: Any
    fact_check_result: Any
    quality_check_result: Any
    duplicate_check_result: Any
    evaluation_result: Any
    current_status: str
    retry_count: int
    max_retries: int
    revision_feedback: list[str]
    errors: list[str]
    warnings: list[str]
    agent_logs: list[dict[str, Any]]