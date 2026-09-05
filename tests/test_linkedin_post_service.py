"""Deterministic tests for draft-only LinkedIn post generation."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from database.database import Base
from database.models import Topic
from services.llm_service import LLMProviderError, LLMService, MockLLMProvider
from services.research_service import ResearchArticle
from services.topic_selection_service import TopicCandidate, TopicSelectionResult
from services.linkedin_post_service import (
    LinkedInPost,
    LinkedInPostService,
    PostGenerationError,
    PostMetrics,
)


def make_topic() -> TopicCandidate:
    return TopicCandidate(
        title="New Multimodal AI Model",
        summary="A research model combines language and vision capabilities.",
        source="Example AI Research",
        url="https://example.com/article",
        category="Multimodal AI",
        publication_date=datetime.now(timezone.utc),
        importance_score=8,
        freshness_score=9,
        linkedin_score=8,
        technical_relevance=8,
        llm_confidence=0.8,
        audience_value=8,
        novelty_score=10,
        selection_score=8,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        linkedin_post_min_characters=100,
        linkedin_post_max_characters=3000,
        linkedin_max_hashtags=5,
    )


@pytest.fixture
def service(settings: Settings) -> LinkedInPostService:
    return LinkedInPostService(
        llm_service=LLMService(provider=MockLLMProvider(), settings=settings),
        settings=settings,
    )


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as database_session:
        yield database_session
    engine.dispose()


def test_service_initialization(service: LinkedInPostService) -> None:
    assert isinstance(service, LinkedInPostService)
    assert isinstance(service.llm_service.provider, MockLLMProvider)


def test_valid_topic_input_and_selection_result(service: LinkedInPostService) -> None:
    result = TopicSelectionResult(selected_topic=make_topic())

    post = service.generate_post(result)

    assert isinstance(post, LinkedInPost)
    assert post.topic_category == "Multimodal AI"


def test_post_generation_contains_all_structured_sections(
    service: LinkedInPostService,
) -> None:
    post = service.generate_post(make_topic())

    assert post.hook
    assert post.body
    assert post.key_insights
    assert post.practical_impact
    assert post.engagement_question.endswith("?")
    assert post.content
    assert post.status == "draft"


def test_source_information_is_preserved(service: LinkedInPostService) -> None:
    post = service.generate_post(make_topic())

    assert post.source_url == "https://example.com/article"
    assert post.source_name == "Example AI Research"
    assert post.topic_category == "Multimodal AI"


def test_hashtags_are_relevant_and_category_dependent(service: LinkedInPostService) -> None:
    multimodal = service.generate_hashtags("Multimodal AI", "Model")
    robotics = service.generate_hashtags("Robotics", "Robot")

    assert multimodal != robotics
    assert 3 <= len(multimodal) <= 5
    assert all(tag.startswith("#") and " " not in tag for tag in multimodal)


def test_invalid_and_duplicate_hashtags_are_rejected() -> None:
    values = {
        "title": "Title",
        "hook": "Hook",
        "body": "Body",
        "key_insights": ["Insight"],
        "practical_impact": "Impact",
        "engagement_question": "A question?",
        "source_url": "https://example.com",
        "source_name": "Example",
        "topic_category": "AI",
    }
    with pytest.raises(ValidationError):
        LinkedInPost(**values, hashtags=["#AI", "bad tag"])
    with pytest.raises(ValidationError):
        LinkedInPost(**values, hashtags=["#AI", "#ai"])


def test_post_metrics_are_deterministic(service: LinkedInPostService) -> None:
    post = service.generate_post(make_topic())
    first = service.calculate_post_metrics(post.content)
    second = service.calculate_post_metrics(post.content)

    assert first == second
    assert isinstance(first, PostMetrics)
    assert first.character_count == len(post.content)
    assert first.word_count > 0
    assert first.paragraph_count >= 4
    assert first.hashtag_count == len(post.hashtags)
    assert first.has_engagement_question
    assert first.has_hook


def test_post_length_validation(service: LinkedInPostService) -> None:
    post = service.generate_post(make_topic())
    short_settings = Settings(linkedin_post_min_characters=5000)
    short_service = LinkedInPostService(
        llm_service=service.llm_service, settings=short_settings
    )

    with pytest.raises(PostGenerationError, match="shorter"):
        short_service.validate_post(post)


def test_content_safety_rejects_secret_and_placeholder(service: LinkedInPostService) -> None:
    post = service.generate_post(make_topic())
    secret_post = post.model_copy(update={"content": "GEMINI_API_KEY=secret"})
    placeholder_post = post.model_copy(update={"content": "lorem ipsum " * 30})

    with pytest.raises(PostGenerationError, match="secret"):
        service.validate_post(secret_post)
    with pytest.raises(PostGenerationError, match="placeholder"):
        service.validate_post(placeholder_post)


def test_empty_topic_is_rejected(service: LinkedInPostService) -> None:
    empty = TopicCandidate.model_construct(
        title="", summary="", source="Example", url="https://example.com", category="AI"
    )

    with pytest.raises(ValueError, match="title and summary"):
        service.generate_post(empty)


def test_missing_selected_topic_is_rejected(service: LinkedInPostService) -> None:
    with pytest.raises(ValueError, match="selected topic"):
        service.generate_post(TopicSelectionResult())


class FailingLLM:
    def generate_structured(self, prompt: str, response_model: type[LinkedInPost]) -> LinkedInPost:
        raise LLMProviderError("provider failed")


class MalformedLLM:
    def generate_structured(self, prompt: str, response_model: type[LinkedInPost]) -> object:
        return {"not": "a post"}


def test_llm_failure_is_wrapped(service: LinkedInPostService) -> None:
    failing_service = LinkedInPostService(llm_service=FailingLLM(), settings=service.settings)

    with pytest.raises(PostGenerationError, match="Unable to generate"):
        failing_service.generate_post(make_topic())


def test_malformed_llm_response_is_rejected(service: LinkedInPostService) -> None:
    malformed_service = LinkedInPostService(llm_service=MalformedLLM(), settings=service.settings)

    with pytest.raises(PostGenerationError, match="invalid"):
        malformed_service.generate_post(make_topic())


def test_database_draft_storage_and_status(
    service: LinkedInPostService, session: Session
) -> None:
    topic = make_topic()
    session.add(
        Topic(
            title=topic.title,
            summary=topic.summary,
            source=topic.source,
            source_url=topic.url,
            category=topic.category,
        )
    )
    session.commit()
    post = service.generate_post(topic)

    stored = service.save_draft(session, post, topic)

    assert stored.status == "draft"
    assert stored.content == post.content
    assert stored.published_at is None
    assert stored.linkedin_post_id is None


def test_draft_requires_existing_database_topic(
    service: LinkedInPostService, session: Session
) -> None:
    with pytest.raises(ValueError, match="already exist"):
        service.save_draft(session, service.generate_post(make_topic()), make_topic())


def test_no_publishing_behavior(service: LinkedInPostService) -> None:
    assert not hasattr(service, "publish_post")
    assert not hasattr(service, "publish")


def test_generated_post_is_draft_only(service: LinkedInPostService) -> None:
    post = service.generate_post(make_topic())

    assert post.status == "draft"
    assert "published" not in post.content.lower()


def test_post_validation_is_repeatable(service: LinkedInPostService) -> None:
    post = service.generate_post(make_topic())

    assert service.validate_post(post)
    assert service.validate_post(post)


def test_research_article_can_be_converted_to_candidate(service: LinkedInPostService) -> None:
    article = ResearchArticle(
        title="New AI agent workflow",
        summary="An AI agent improves a technical workflow.",
        source="Example",
        source_url="https://example.com/agent",
        category="AI Agents",
    )
    candidate = TopicCandidate(
        title=article.title,
        summary=article.summary,
        source=article.source,
        url=article.source_url,
        category=article.category,
        importance_score=7,
        freshness_score=8,
        linkedin_score=7,
        technical_relevance=7,
        llm_confidence=0.8,
        audience_value=7,
        novelty_score=10,
        selection_score=7,
    )

    post = service.generate_post(candidate)

    assert post.topic_category == "AI Agents"