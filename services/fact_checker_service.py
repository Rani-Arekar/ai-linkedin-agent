"""Source-grounded, deterministic fact checking for LinkedIn drafts."""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Iterable

from pydantic import BaseModel, Field

from services.linkedin_post_service import LinkedInPost
from services.llm_service import ArticleAnalysis, LLMService, LLMServiceError
from services.research_service import ResearchArticle

logger = logging.getLogger(__name__)


class ClaimStatus(StrEnum):
    """Evidence status assigned to a generated claim."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNCERTAIN = "UNCERTAIN"


class FactCheckStatus(StrEnum):
    """Overall fact-check decision."""

    PASS = "PASS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAIL = "FAIL"


class ClaimCheck(BaseModel):
    """Evidence assessment for one extracted factual claim."""

    claim: str = Field(min_length=1, max_length=1_000)
    status: ClaimStatus
    evidence: str = Field(default="", max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1_000)


class FactCheckResult(BaseModel):
    """Structured result of source-grounded claim verification."""

    overall_status: FactCheckStatus
    overall_score: float = Field(ge=0, le=10)
    claims: list[ClaimCheck] = Field(default_factory=list)
    supported_claims: int = Field(ge=0)
    unsupported_claims: int = Field(ge=0)
    uncertain_claims: int = Field(ge=0)
    hallucination_risk: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1, max_length=2_000)
    recommendations: list[str] = Field(default_factory=list, max_length=10)


class FactCheckerService:
    """Check draft claims against only the supplied article and analysis."""

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self.llm_service = llm_service

    def extract_claims(self, post: LinkedInPost) -> list[str]:
        """Extract sentence-like factual statements, excluding questions and labels."""

        claims: list[str] = []
        text = "\n".join((post.body, *post.key_insights))
        for sentence in re.split(r"(?<=[.!])\s+|\n+", text):
            cleaned = re.sub(r"^[-•*]\s*", "", sentence).strip()
            if not cleaned or cleaned.endswith("?") or len(cleaned.split()) < 4:
                continue
            lowered = cleaned.lower()
            if lowered.startswith(("key insights", "practical impact")):
                continue
            if lowered.startswith(("i think", "in my view", "we should", "this could")):
                continue
            if cleaned not in claims:
                claims.append(cleaned)
        return claims

    def check_post(
        self,
        post: LinkedInPost,
        article: ResearchArticle,
        analysis: ArticleAnalysis | None = None,
    ) -> FactCheckResult:
        """Evaluate all extracted claims without browsing or using outside knowledge."""

        if not post.content.strip():
            return self._failure("The post is empty.", "Provide a non-empty draft.")
        if not article.title.strip() and not article.summary.strip():
            return self._failure("No source evidence was supplied.", "Provide source content before fact checking.")
        if analysis is None and self.llm_service is not None:
            try:
                analysis = self.llm_service.analyze_article(article)
            except LLMServiceError as error:
                logger.warning("LLM evidence enrichment unavailable: %s", type(error).__name__)
        evidence = self._evidence(article, analysis)
        checks = [self._check_claim(claim, evidence) for claim in self.extract_claims(post)]
        supported = sum(check.status == ClaimStatus.SUPPORTED for check in checks)
        unsupported = sum(check.status in {ClaimStatus.UNSUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED} for check in checks)
        uncertain = sum(check.status == ClaimStatus.UNCERTAIN for check in checks)
        risk = min(1.0, (unsupported * 0.35) + (uncertain * 0.18) + (0.15 if not evidence.strip() else 0))
        if unsupported or not checks:
            status = FactCheckStatus.FAIL if unsupported >= 2 or risk >= 0.7 else FactCheckStatus.NEEDS_REVIEW
        elif uncertain:
            status = FactCheckStatus.NEEDS_REVIEW
        else:
            status = FactCheckStatus.PASS
        score = round(max(0.0, 10.0 - unsupported * 3.5 - uncertain * 1.5), 2)
        recommendations = [
            f"Review or remove unsupported claim: {check.claim}"
            for check in checks
            if check.status in {ClaimStatus.UNSUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED, ClaimStatus.UNCERTAIN}
        ]
        return FactCheckResult(
            overall_status=status,
            overall_score=score,
            claims=checks,
            supported_claims=supported,
            unsupported_claims=unsupported,
            uncertain_claims=uncertain,
            hallucination_risk=round(risk, 2),
            explanation="Claims were compared only with the supplied article evidence.",
            recommendations=recommendations,
        )

    @staticmethod
    def _evidence(article: ResearchArticle, analysis: ArticleAnalysis | None) -> str:
        parts = [article.title, article.summary, article.source, article.category]
        if analysis is not None:
            parts.extend([analysis.summary, analysis.why_it_matters, *analysis.key_points, *analysis.technical_concepts])
        return " ".join(part for part in parts if part).lower()

    @staticmethod
    def _check_claim(claim: str, evidence: str) -> ClaimCheck:
        claim_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim)
        evidence_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", evidence))
        if claim_numbers and not all(number in evidence_numbers for number in claim_numbers):
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.UNSUPPORTED,
                evidence=evidence[:2_000],
                confidence=0.98,
                reason="A numerical value in the claim is absent from the supplied evidence.",
            )
        tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
        evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence))
        overlap = len(tokens & evidence_tokens) / max(len(tokens), 1)
        stop_words = {"a", "an", "and", "are", "for", "in", "is", "it", "of", "on", "the", "this", "to", "with"}
        missing_content = {token for token in tokens - evidence_tokens if token not in stop_words}
        if overlap >= 0.58 and not missing_content:
            status, confidence, reason = ClaimStatus.SUPPORTED, 0.9, "The claim's key terms are directly present in the supplied evidence."
        elif overlap >= 0.30:
            status, confidence, reason = ClaimStatus.PARTIALLY_SUPPORTED, 0.6, "Some claim terms match the evidence, but the complete claim is not established."
        else:
            status, confidence, reason = ClaimStatus.UNCERTAIN, 0.35, "The supplied evidence is insufficient to verify this claim."
        return ClaimCheck(claim=claim, status=status, evidence=evidence[:2_000], confidence=confidence, reason=reason)

    @staticmethod
    def _failure(explanation: str, recommendation: str) -> FactCheckResult:
        return FactCheckResult(
            overall_status=FactCheckStatus.FAIL,
            overall_score=0,
            hallucination_risk=1,
            supported_claims=0,
            unsupported_claims=0,
            uncertain_claims=0,
            explanation=explanation,
            recommendations=[recommendation],
        )