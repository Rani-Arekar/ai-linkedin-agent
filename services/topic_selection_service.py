"""Deterministic, LLM-enriched selection of the best research topic."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database.models import Topic
from services.llm_service import ArticleAnalysis, LLMService, LLMServiceError
from services.research_service import ResearchArticle

logger = logging.getLogger(__name__)


class TopicCandidate(BaseModel):
    """A validated article enriched with selection metrics and reasoning."""

    article_id: int | None = None
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(default="", max_length=10_000)
    source: str = Field(min_length=1, max_length=255)
    url: str = Field(min_length=1, max_length=2048)
    category: str = Field(min_length=1, max_length=100)
    publication_date: datetime | None = None
    importance_score: float = Field(ge=0, le=10)
    freshness_score: float = Field(ge=0, le=10)
    linkedin_score: float = Field(ge=0, le=10)
    technical_relevance: float = Field(ge=0, le=10)
    llm_confidence: float = Field(ge=0, le=1)
    audience_value: float = Field(ge=0, le=10)
    novelty_score: float = Field(ge=0, le=10)
    selection_score: float = Field(ge=0, le=10)
    selection_reasons: list[str] = Field(default_factory=list, max_length=10)
    llm_analysis: ArticleAnalysis | None = None


class TopicSelectionResult(BaseModel):
    """Ranked topic candidates and the deterministic recommendation."""

    selected_topic: TopicCandidate | None = None
    ranked_candidates: list[TopicCandidate] = Field(default_factory=list)
    selection_score: float | None = Field(default=None, ge=0, le=10)
    selection_reasons: list[str] = Field(default_factory=list)


class TopicSelectionService:
    """Build, score, rank, and select research topics using explainable rules."""

    def __init__(
        self,
        llm_service: LLMService | Any | None = None,
        settings: Settings | None = None,
        freshness_days: int = 30,
    ) -> None:
        self.settings = settings or get_settings()
        if llm_service is None:
            self.llm_service = LLMService(settings=self.settings)
        elif hasattr(llm_service, "analyze_article"):
            self.llm_service = llm_service
        else:
            self.llm_service = LLMService(provider=llm_service, settings=self.settings)
        self.freshness_days = freshness_days

    def build_candidates(
        self,
        articles: Iterable[ResearchArticle | Any],
        session: Session | None = None,
    ) -> list[TopicCandidate]:
        """Validate articles, exclude history, enrich them, and return scoreable candidates."""

        history = self._history(session) if session is not None else []
        candidates: list[TopicCandidate] = []
        for article in articles:
            try:
                article = ResearchArticle.model_validate(article)
                if not self._eligible(article, history):
                    continue
                analysis = self._analyze(article)
                candidates.append(self._candidate(article, analysis, history))
            except (ValidationError, ValueError) as error:
                logger.warning("Invalid topic candidate skipped: %s", type(error).__name__)
            except Exception:
                logger.exception("Unexpected topic candidate failure")
        return candidates

    def score_candidate(
        self,
        candidate: TopicCandidate,
        recent_categories: Sequence[str] = (),
    ) -> TopicCandidate:
        """Recalculate the bounded weighted score and evidence-based explanation."""

        diversity_bonus = self._diversity_score(candidate.category, recent_categories)
        weighted = (
            candidate.importance_score * 0.20
            + candidate.freshness_score * self.settings.topic_selection_freshness_weight
            + candidate.linkedin_score * 0.20
            + candidate.technical_relevance * 0.15
            + candidate.llm_confidence * 10 * 0.10
            + candidate.audience_value * 0.10
            + candidate.novelty_score * 0.10
            + diversity_bonus * self.settings.topic_selection_diversity_weight
        )
        total_weight = (
            0.20
            + self.settings.topic_selection_freshness_weight
            + 0.20
            + 0.15
            + 0.10
            + 0.10
            + 0.10
            + self.settings.topic_selection_diversity_weight
        )
        score = round(min(10.0, max(0.0, weighted / total_weight * 1.0)), 2)
        reasons = self._reasons(candidate, recent_categories, diversity_bonus)
        return candidate.model_copy(update={"selection_score": score, "selection_reasons": reasons})

    def rank_candidates(
        self,
        candidates: Iterable[TopicCandidate],
        recent_categories: Sequence[str] = (),
    ) -> list[TopicCandidate]:
        """Score and sort candidates deterministically, highest score first."""

        scored = [self.score_candidate(candidate, recent_categories) for candidate in candidates]
        scored.sort(key=lambda item: (-item.selection_score, item.title.lower(), item.url))
        return scored[: self.settings.topic_selection_max_candidates]

    def select_best_topic(
        self,
        articles: Iterable[ResearchArticle | Any],
        session: Session | None = None,
        recent_categories: Sequence[str] = (),
    ) -> TopicSelectionResult:
        """Return a ranked result, or an explicit empty result when no topic qualifies."""

        candidates = self.build_candidates(articles, session)
        ranked = self.rank_candidates(candidates, recent_categories)
        eligible = [
            candidate
            for candidate in ranked
            if candidate.selection_score >= self.settings.topic_selection_min_score
        ]
        selected = eligible[0] if eligible else None
        return TopicSelectionResult(
            selected_topic=selected,
            ranked_candidates=ranked,
            selection_score=selected.selection_score if selected else None,
            selection_reasons=selected.selection_reasons if selected else [],
        )

    def _candidate(
        self,
        article: ResearchArticle,
        analysis: ArticleAnalysis | None,
        history: Sequence[Topic],
    ) -> TopicCandidate:
        technical_relevance = self._technical_relevance(article, analysis)
        importance_score, freshness_score, linkedin_score = self._research_scores(article)
        audience_value = analysis.linkedin_relevance if analysis else linkedin_score
        novelty = self._novelty(article, history)
        base = TopicCandidate(
            title=article.title,
            summary=article.summary,
            source=article.source,
            url=article.source_url,
            category=article.category,
            publication_date=article.published_at,
            importance_score=importance_score,
            freshness_score=freshness_score,
            linkedin_score=linkedin_score,
            technical_relevance=technical_relevance,
            llm_confidence=analysis.confidence if analysis else 0.0,
            audience_value=bounded(audience_value),
            novelty_score=novelty,
            selection_score=0.0,
            llm_analysis=analysis,
        )
        return self.score_candidate(base)

    def _analyze(self, article: ResearchArticle) -> ArticleAnalysis | None:
        """Try LLM enrichment per article; deterministic selection continues on failure."""

        try:
            return self.llm_service.analyze_article(article)
        except LLMServiceError as error:
            logger.warning("LLM analysis unavailable for topic: %s", type(error).__name__)
            return None
        except Exception:
            logger.exception("Unexpected LLM analysis failure")
            return None

    def _eligible(self, article: ResearchArticle, history: Sequence[Topic]) -> bool:
        if not article.title.strip() or not article.summary.strip():
            return False
        if article.published_at and article.published_at < datetime.now(timezone.utc) - timedelta(days=self.freshness_days):
            logger.info("Stale topic skipped: %s", article.title)
            return False
        normalized_url = normalize_url(article.source_url)
        normalized_title = normalize_title(article.title)
        return not any(
            normalized_url == normalize_url(topic.source_url)
            or normalized_title == normalize_title(topic.title)
            for topic in history
        )

    @staticmethod
    def _history(session: Session | None) -> list[Topic]:
        return list(session.scalars(select(Topic))) if session is not None else []

    @staticmethod
    def _technical_relevance(article: ResearchArticle, analysis: ArticleAnalysis | None) -> float:
        if analysis is not None:
            return bounded(5.0 + min(len(analysis.technical_concepts), 5))
        return 8.0 if article.category != "General AI" else 5.0

    @staticmethod
    def _research_scores(article: ResearchArticle) -> tuple[float, float, float]:
        """Read optional Phase 3 score extensions, with transparent fallbacks."""

        importance = getattr(article, "importance_score", None)
        freshness = getattr(article, "freshness_score", None)
        linkedin = getattr(article, "linkedin_score", None)
        importance_score = article_score(importance if importance is not None else (7.0 if article.category != "General AI" else 5.0))
        if freshness is None:
            age_days = (
                max((datetime.now(timezone.utc) - article.published_at).total_seconds() / 86400, 0)
                if article.published_at
                else 31
            )
            freshness = 10.0 if age_days <= 1 else 8.0 if age_days <= 3 else 6.0 if age_days <= 7 else 3.0 if age_days <= 30 else 1.0
        freshness_score = article_score(freshness)
        linkedin_score = article_score(linkedin if linkedin is not None else (importance_score + freshness_score) / 2)
        return importance_score, freshness_score, linkedin_score

    @staticmethod
    def _novelty(article: ResearchArticle, history: Sequence[Topic]) -> float:
        if not history:
            return 10.0
        title = normalize_title(article.title)
        max_similarity = max(
            (SequenceMatcher(None, title, normalize_title(topic.title)).ratio() for topic in history),
            default=0.0,
        )
        category_seen = sum(topic.category == article.category for topic in history)
        return bounded(10.0 - max_similarity * 7.0 - min(category_seen, 3) * 0.5)

    @staticmethod
    def _diversity_score(category: str, recent_categories: Sequence[str]) -> float:
        if not recent_categories:
            return 5.0
        count = Counter(recent_categories).get(category, 0)
        return bounded(10.0 - count * 3.0)

    @staticmethod
    def _reasons(
        candidate: TopicCandidate,
        recent_categories: Sequence[str],
        diversity_bonus: float,
    ) -> list[str]:
        reasons: list[str] = []
        if candidate.importance_score >= 7:
            reasons.append("High technical importance")
        if candidate.freshness_score >= 8:
            reasons.append("Very recent")
        if candidate.linkedin_score >= 7:
            reasons.append("Strong LinkedIn potential")
        if candidate.technical_relevance >= 7:
            reasons.append("Relevant to AI professionals")
        if candidate.audience_value >= 7:
            reasons.append("High audience value")
        if candidate.novelty_score >= 7:
            reasons.append("Not recently covered")
        if recent_categories and diversity_bonus >= 7:
            reasons.append("Adds topic diversity")
        return reasons or ["Best available score among eligible candidates"]


def bounded(value: float) -> float:
    """Clamp a metric to the service's 0-10 scale."""

    return round(min(10.0, max(0.0, float(value))), 2)


def article_score(value: float | None) -> float:
    """Convert missing research scores to a neutral deterministic value."""

    return bounded(value if value is not None else 5.0)


def normalize_url(value: str) -> str:
    """Normalize URL text for historical duplicate checks."""

    return value.strip().lower().rstrip("/")


def normalize_title(value: str) -> str:
    """Normalize title text for exact historical duplicate checks."""

    return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()


if __name__ == "__main__":
    print("Topic selection service is ready; provide ResearchArticle values to select_best_topic().")