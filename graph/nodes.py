"""Thin LangGraph node adapters around the existing application services."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import crud
from database.models import Post
from graph.state import WorkflowState
from services.fact_checker_service import FactCheckerService
from services.linkedin_post_service import LinkedInPost, LinkedInPostService
from services.quality_checker_service import QualityCheckerService
from services.research_service import ResearchArticle, ResearchService
from services.topic_selection_service import TopicCandidate, TopicSelectionService

logger = logging.getLogger(__name__)


class DuplicateCheckResult(BaseModel):
    """Deterministic post-content duplicate check result."""

    is_duplicate: bool
    similarity_score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    status: str = Field(min_length=1)


@dataclass(slots=True)
class WorkflowDependencies:
    """Services supplied to graph nodes, keeping runtime objects out of state."""

    research: ResearchService
    topic_selection: TopicSelectionService
    writer: LinkedInPostService
    fact_checker: FactCheckerService
    quality_checker: QualityCheckerService
    session: Session | None = None
    enable_duplicate_check: bool = True


def _record(state: WorkflowState, agent: str, status: str, error: str | None = None) -> dict[str, Any]:
    log = {"agent_name": agent, "status": status, "created_at": datetime.now(timezone.utc).isoformat()}
    updates: dict[str, Any] = {"agent_logs": [*state.get("agent_logs", []), log]}
    if error:
        updates["errors"] = [*state.get("errors", []), error]
    return updates


def research_agent(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Fetch research through the Phase 3 service and place articles in state."""

    try:
        articles, failures = dependencies.research.fetch_articles()
        updates: dict[str, Any] = {"research_articles": articles, "warnings": [*state.get("warnings", []), *failures]}
        updates.update(_record(state, "research", "success" if articles else "empty"))
        if not articles:
            updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]
    except Exception as error:
        logger.warning("Research node failed: %s", type(error).__name__)
        updates = _record(state, "research", "failed", "Research failed")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]


def topic_selection_agent(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Select a topic through the Phase 5 service without duplicating its logic."""

    try:
        result = dependencies.topic_selection.select_best_topic(
            state.get("research_articles", []), dependencies.session
        )
        updates: dict[str, Any] = {"selected_topic": result.selected_topic, "evaluation_result": result}
        updates.update(_record(state, "topic_selection", "success" if result.selected_topic else "empty"))
        if result.selected_topic is None:
            updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]
    except Exception as error:
        logger.warning("Topic selection node failed: %s", type(error).__name__)
        updates = _record(state, "topic_selection", "failed", "Topic selection failed")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]


def linkedin_writer_agent(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Generate a draft through the Phase 6 service, including revision feedback."""

    topic = state.get("selected_topic")
    if topic is None:
        updates = _record(state, "writer", "failed", "Writer requires a selected topic")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]
    try:
        post = dependencies.writer.generate_post(
            topic, state.get("revision_feedback", [])
        )
        updates: dict[str, Any] = {"generated_post": post, "current_status": "DRAFT_GENERATED"}
        updates.update(_record(state, "writer", "success"))
        return updates  # type: ignore[return-value]
    except Exception as error:
        logger.warning("Writer node failed: %s", type(error).__name__)
        updates = _record(state, "writer", "failed", "Writer failed")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]


def fact_checker_agent(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Run the existing Phase 7 fact checker against the selected source article."""

    post = state.get("generated_post")
    topic = state.get("selected_topic")
    if post is None or topic is None:
        updates = _record(state, "fact_checker", "failed", "Fact checker requires a draft and topic")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]
    try:
        article = _article_from_topic(topic)
        result = dependencies.fact_checker.check_post(post, article, topic.llm_analysis)
        updates: dict[str, Any] = {"fact_check_result": result}
        updates.update(_record(state, "fact_checker", "success"))
        return updates  # type: ignore[return-value]
    except Exception as error:
        logger.warning("Fact checker node failed: %s", type(error).__name__)
        updates = _record(state, "fact_checker", "failed", "Fact checker failed")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]


def quality_checker_agent(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Run the existing Phase 7 quality checker."""

    post = state.get("generated_post")
    if post is None:
        updates = _record(state, "quality_checker", "failed", "Quality checker requires a draft")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]
    try:
        result = dependencies.quality_checker.check_post(post)
        updates: dict[str, Any] = {"quality_check_result": result}
        updates.update(_record(state, "quality_checker", "success"))
        return updates  # type: ignore[return-value]
    except Exception as error:
        logger.warning("Quality checker node failed: %s", type(error).__name__)
        updates = _record(state, "quality_checker", "failed", "Quality checker failed")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]


def duplicate_detection_agent(state: WorkflowState, dependencies: WorkflowDependencies) -> WorkflowState:
    """Compare draft text with stored posts using lightweight title-free similarity."""

    post = state.get("generated_post")
    if post is None:
        updates = _record(state, "duplicate_detection", "failed", "Duplicate check requires a draft")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]
    try:
        if not dependencies.enable_duplicate_check or dependencies.session is None:
            result = DuplicateCheckResult(is_duplicate=False, similarity_score=0, reason="Duplicate check disabled", status="SKIPPED")
        else:
            normalized = normalize_content(post.content)
            similarities = [
                SequenceMatcher(None, normalized, normalize_content(previous.content)).ratio()
                for previous in dependencies.session.scalars(select(Post).where(Post.status != "rejected"))
            ]
            score = max(similarities, default=0.0)
            result = DuplicateCheckResult(
                is_duplicate=score >= 0.85,
                similarity_score=round(score, 3),
                reason="Draft is highly similar to stored content" if score >= 0.85 else "No highly similar stored draft found",
                status="DUPLICATE" if score >= 0.85 else "PASS",
            )
        updates: dict[str, Any] = {"duplicate_check_result": result}
        updates.update(_record(state, "duplicate_detection", "success"))
        return updates  # type: ignore[return-value]
    except Exception as error:
        logger.warning("Duplicate node failed: %s", type(error).__name__)
        updates = _record(state, "duplicate_detection", "failed", "Duplicate detection failed")
        updates["current_status"] = "FAILED"
        return updates  # type: ignore[return-value]


def _article_from_topic(topic: TopicCandidate) -> ResearchArticle:
    return ResearchArticle(
        title=topic.title,
        summary=topic.summary,
        source=topic.source,
        source_url=topic.url,
        category=topic.category,
        published_at=topic.publication_date,
    )


def normalize_content(value: str) -> str:
    """Normalize content for lightweight duplicate comparison."""

    return re.sub(r"\s+", " ", value.lower()).strip()