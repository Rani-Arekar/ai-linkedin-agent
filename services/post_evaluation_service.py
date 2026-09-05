"""Coordinator for fact checking and deterministic quality approval gates."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from services.fact_checker_service import FactCheckResult, FactCheckStatus, FactCheckerService
from services.linkedin_post_service import LinkedInPost
from services.quality_checker_service import QualityCheckResult, QualityCheckerService, QualityStatus
from services.research_service import ResearchArticle
from services.llm_service import ArticleAnalysis


class EvaluationStatus(StrEnum):
    """Final deterministic post evaluation state."""

    APPROVED = "APPROVED"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECTED = "REJECTED"


class PostEvaluationResult(BaseModel):
    """Combined fact and quality evaluation with blocking issues."""

    fact_check: FactCheckResult
    quality_check: QualityCheckResult
    overall_score: float = Field(ge=0, le=10)
    final_status: EvaluationStatus
    blocking_issues: list[str] = Field(default_factory=list, max_length=20)
    recommendations: list[str] = Field(default_factory=list, max_length=20)


class PostEvaluationService:
    """Run fact and quality checks; deterministic gates cannot be overridden by LLM output."""

    def __init__(
        self,
        fact_checker: FactCheckerService | None = None,
        quality_checker: QualityCheckerService | None = None,
    ) -> None:
        self.fact_checker = fact_checker or FactCheckerService()
        self.quality_checker = quality_checker or QualityCheckerService()

    def evaluate_post(
        self,
        post: LinkedInPost,
        article: ResearchArticle,
        analysis: ArticleAnalysis | None = None,
    ) -> PostEvaluationResult:
        """Return a review decision and recommendations without modifying the draft."""

        fact_check = self.fact_checker.check_post(post, article, analysis)
        quality_check = self.quality_checker.check_post(post)
        blocking: list[str] = []
        if quality_check.status == QualityStatus.FAIL:
            blocking.extend(quality_check.issues)
        if fact_check.overall_status == FactCheckStatus.FAIL:
            blocking.extend(fact.reason for fact in fact_check.claims if fact.status.value in {"UNSUPPORTED", "PARTIALLY_SUPPORTED"})
        if fact_check.hallucination_risk >= 0.7:
            blocking.append("High project-level hallucination risk.")
        overall = round(fact_check.overall_score * 0.55 + quality_check.overall_score * 0.45, 2)
        if blocking:
            status = EvaluationStatus.REJECTED
        elif fact_check.overall_status == FactCheckStatus.NEEDS_REVIEW or quality_check.status == QualityStatus.NEEDS_REVIEW:
            status = EvaluationStatus.NEEDS_REVISION
        else:
            status = EvaluationStatus.APPROVED
        return PostEvaluationResult(
            fact_check=fact_check,
            quality_check=quality_check,
            overall_score=overall,
            final_status=status,
            blocking_issues=blocking,
            recommendations=[*fact_check.recommendations, *quality_check.recommendations],
        )