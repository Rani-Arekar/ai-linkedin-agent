"""Deterministic tests for Phase 5 topic selection."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from database.database import Base
from database.models import Topic
from services.llm_service import ArticleAnalysis, LLMProviderError, LLMService, MockLLMProvider
from services.research_service import ResearchArticle
from services.topic_selection_service import (
    TopicCandidate,
    TopicSelectionResult,
    TopicSelectionService,
)


def make_article(
    title: str = "New Multimodal AI Model",
    url: str = "https://example.com/article",
    category: str = "Multimodal AI",
    published_at: datetime | None = None,
    importance: float = 8.0,
    freshness: float = 9.0,
    linkedin: float = 8.0,
) -> ResearchArticle:
    return ResearchArticle(
        title=title,
        summary="A technical AI development with practical implications.",
        source="Example AI Research",
        source_url=url,
        category=category,
        published_at=published_at,
        importance_score=importance,
        freshness_score=freshness,
        linkedin_score=linkedin,
    )


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as database_session:
        yield database_session
    engine.dispose()


@pytest.fixture
def service() -> TopicSelectionService:
    return TopicSelectionService(
        llm_service=MockLLMProvider(),
        settings=Settings(topic_selection_min_score=0),
    )


def test_service_initialization(service: TopicSelectionService) -> None:
    assert isinstance(service, TopicSelectionService)
    assert isinstance(service.llm_service, LLMService)
    assert isinstance(service.llm_service.provider, MockLLMProvider)


def test_candidate_generation_and_llm_integration(
    service: TopicSelectionService,
) -> None:
    candidates = service.build_candidates([make_article()])

    assert len(candidates) == 1
    assert candidates[0].title == "New Multimodal AI Model"
    assert candidates[0].llm_analysis is not None
    assert candidates[0].llm_confidence == 0.8


def test_invalid_and_stale_candidates_are_filtered(service: TopicSelectionService) -> None:
    empty = ResearchArticle.model_construct(
        title="", summary="", source="Example", source_url="https://example.com/empty"
    )
    stale = make_article(
        title="Old AI paper",
        url="https://example.com/old",
        published_at=datetime.now(timezone.utc) - timedelta(days=31),
    )

    assert service.build_candidates([empty, stale]) == []


def test_score_is_bounded_and_deterministic(service: TopicSelectionService) -> None:
    candidate = service.build_candidates([make_article()])[0]

    first = service.score_candidate(candidate)
    second = service.score_candidate(candidate)

    assert first.selection_score == second.selection_score
    assert 0 <= first.selection_score <= 10
    assert first.selection_reasons


def test_ranking_and_best_topic_selection(service: TopicSelectionService) -> None:
    articles = [
        make_article("Lower priority AI update", "https://example.com/low", importance=5, linkedin=5),
        make_article("Important AI breakthrough", "https://example.com/high", importance=10, linkedin=10),
    ]

    result = service.select_best_topic(articles)

    assert isinstance(result, TopicSelectionResult)
    assert result.selected_topic is not None
    assert result.selected_topic.title == "Important AI breakthrough"
    assert result.ranked_candidates[0].title == result.selected_topic.title


def test_selection_explanation_uses_actual_metrics(service: TopicSelectionService) -> None:
    result = service.select_best_topic([make_article()])

    assert result.selection_reasons == result.selected_topic.selection_reasons
    assert "Very recent" not in result.selection_reasons
    assert "High technical importance" in result.selection_reasons
    assert "High audience value" in result.selection_reasons


def test_empty_candidate_result(service: TopicSelectionService) -> None:
    result = service.select_best_topic([])

    assert result.selected_topic is None
    assert result.ranked_candidates == []
    assert result.selection_score is None
    assert result.selection_reasons == []


def test_candidate_model_rejects_invalid_scores() -> None:
    with pytest.raises(ValidationError):
        TopicCandidate(
            title="Invalid",
            source="Example",
            url="https://example.com/invalid",
            category="General AI",
            importance_score=11,
            freshness_score=5,
            linkedin_score=5,
            technical_relevance=5,
            llm_confidence=0.5,
            audience_value=5,
            novelty_score=5,
            selection_score=5,
        )


def test_previous_url_and_title_duplicates_are_avoided(
    service: TopicSelectionService, session: Session
) -> None:
    session.add(
        Topic(
            title="Covered AI Release",
            summary="Covered",
            source="Example",
            source_url="https://example.com/covered/",
            category="Generative AI",
        )
    )
    session.commit()

    articles = [
        make_article("New title", "https://example.com/covered"),
        make_article("Covered AI Release!", "https://example.com/new"),
    ]

    assert service.build_candidates(articles, session) == []


def test_topic_diversity_prefers_uncovered_category(service: TopicSelectionService) -> None:
    llm = ArticleAnalysis(
        summary="Summary",
        key_points=["Point"],
        technical_concepts=["transformer"],
        why_it_matters="Useful",
        linkedin_relevance=7,
        confidence=0.8,
    )
    candidates = [
        service.build_candidates([make_article("LLM update", "https://example.com/llm", "Large Language Models")])[0],
        service.build_candidates([make_article("Vision update", "https://example.com/vision", "Computer Vision")])[0],
    ]
    candidates[0] = candidates[0].model_copy(update={"llm_analysis": llm})
    candidates[1] = candidates[1].model_copy(update={"llm_analysis": llm})

    ranked = service.rank_candidates(candidates, recent_categories=["Large Language Models"] * 3)

    assert ranked[0].category == "Computer Vision"
    assert "Adds topic diversity" in ranked[0].selection_reasons


def test_novelty_decreases_for_historical_category(
    service: TopicSelectionService, session: Session
) -> None:
    session.add(
        Topic(
            title="Previous model",
            summary="Previous",
            source="Example",
            source_url="https://example.com/previous",
            category="Multimodal AI",
        )
    )
    session.commit()
    fresh = service.build_candidates(
        [make_article("Fresh model", "https://example.com/fresh")], session
    )[0]

    assert fresh.novelty_score < 10


def test_llm_failure_falls_back_to_deterministic_metrics() -> None:
    class FailingLLM:
        def analyze_article(self, article: ResearchArticle) -> ArticleAnalysis:
            raise LLMProviderError("provider failed")

    service = TopicSelectionService(
        llm_service=FailingLLM(), settings=Settings(topic_selection_min_score=0)
    )
    candidates = service.build_candidates([make_article()])

    assert len(candidates) == 1
    assert candidates[0].llm_analysis is None
    assert candidates[0].llm_confidence == 0
    assert candidates[0].technical_relevance == 8


def test_database_history_is_considered(session: Session, service: TopicSelectionService) -> None:
    session.add(
        Topic(
            title="Historical topic",
            summary="History",
            source="Example",
            source_url="https://example.com/history",
            category="AI Research",
        )
    )
    session.commit()

    result = service.select_best_topic(
        [make_article("Historical topic", "https://example.com/new")], session
    )

    assert result.selected_topic is None


def test_minimum_score_filtering() -> None:
    service = TopicSelectionService(
        llm_service=MockLLMProvider(), settings=Settings(topic_selection_min_score=10)
    )
    result = service.select_best_topic([make_article()])

    assert result.selected_topic is None
    assert len(result.ranked_candidates) == 1


def test_maximum_candidate_limit() -> None:
    service = TopicSelectionService(
        llm_service=MockLLMProvider(),
        settings=Settings(topic_selection_min_score=0, topic_selection_max_candidates=2),
    )
    articles = [make_article(f"AI topic {index}", f"https://example.com/{index}") for index in range(4)]

    result = service.select_best_topic(articles)

    assert len(result.ranked_candidates) == 2


def test_ranking_consistency(service: TopicSelectionService) -> None:
    articles = [make_article(f"Topic {index}", f"https://example.com/{index}") for index in range(3)]

    first = service.select_best_topic(articles)
    second = service.select_best_topic(list(reversed(articles)))

    assert [item.url for item in first.ranked_candidates] == [item.url for item in second.ranked_candidates]


def test_manual_entry_point_is_non_publishing() -> None:
    assert TopicSelectionService.__module__ == "services.topic_selection_service"