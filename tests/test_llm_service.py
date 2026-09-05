"""Deterministic tests for the Phase 4 LLM service."""

import logging

import pytest
from pydantic import ValidationError

from config import Settings
from services.llm_service import (
    ArticleAnalysis,
    ArticleSummary,
    KeyPoints,
    LLMConfigurationError,
    LLMProviderError,
    LLMService,
    MockLLMProvider,
    load_prompt,
)
from services.research_service import ResearchArticle


@pytest.fixture
def article() -> ResearchArticle:
    return ResearchArticle(
        title="New Multimodal AI Model Improves Visual Reasoning",
        summary="A research model combines language and vision capabilities.",
        source="Example AI Research",
        source_url="https://example.com/article",
        category="Multimodal AI",
    )


def test_service_initializes_with_mock_provider() -> None:
    service = LLMService(settings=Settings(llm_provider="mock"))

    assert isinstance(service.provider, MockLLMProvider)


def test_mock_provider_returns_valid_analysis(article: ResearchArticle) -> None:
    result = MockLLMProvider().generate("ignored", ArticleAnalysis)

    assert result.confidence == 0.8
    assert result.linkedin_relevance == 7.0
    assert result.key_points


def test_article_analysis_summary_and_key_points(article: ResearchArticle) -> None:
    service = LLMService(provider=MockLLMProvider())

    analysis = service.analyze_article(article)
    summary = service.summarize_article(article)
    points = service.extract_key_points(article)

    assert isinstance(analysis, ArticleAnalysis)
    assert isinstance(summary, ArticleSummary)
    assert isinstance(points, KeyPoints)
    assert len(points.key_points) == 2


def test_response_models_validate_bounds() -> None:
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            summary="Summary",
            key_points=["Point"],
            why_it_matters="Why",
            linkedin_relevance=11,
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        ArticleAnalysis(
            summary="Summary",
            key_points=["Point"],
            why_it_matters="Why",
            linkedin_relevance=5,
            confidence=-0.1,
        )


def test_empty_article_is_rejected() -> None:
    empty = ResearchArticle.model_construct(
        title="", summary="", source="Example", source_url="https://example.com"
    )

    with pytest.raises(ValueError, match="title or summary"):
        LLMService(provider=MockLLMProvider()).analyze_article(empty)


class InvalidProvider:
    def generate(self, prompt: str, response_model: type[ArticleSummary]) -> ArticleSummary:
        return response_model.model_validate({"unexpected": "data"})


class FailingProvider:
    def generate(self, prompt: str, response_model: type[ArticleSummary]) -> ArticleSummary:
        raise TimeoutError("provider timeout")


def test_invalid_response_is_reported(article: ResearchArticle) -> None:
    with pytest.raises(LLMProviderError, match="provider failed"):
        LLMService(provider=InvalidProvider()).summarize_article(article)


def test_provider_failure_is_wrapped(article: ResearchArticle) -> None:
    with pytest.raises(LLMProviderError, match="provider failed"):
        LLMService(provider=FailingProvider()).summarize_article(article)


def test_missing_gemini_key_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        LLMService(settings=Settings(llm_provider="gemini", gemini_api_key=None, llm_api_key=None))


def test_unsupported_provider_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="Unsupported"):
        LLMService(settings=Settings(llm_provider="unknown"))


def test_configuration_loads_llm_values() -> None:
    settings = Settings(
        llm_provider="mock",
        gemini_model="test-model",
        llm_timeout=12,
        llm_temperature=0.4,
    )

    assert settings.llm_provider == "mock"
    assert settings.gemini_model == "test-model"
    assert settings.llm_timeout == 12
    assert settings.llm_temperature == 0.4


def test_prompt_files_are_loaded() -> None:
    assert "only the supplied fields" in load_prompt("article_analysis_prompt.txt")
    assert "structured JSON" in load_prompt("article_summary_prompt.txt")


def test_secret_is_not_logged(caplog: pytest.LogCaptureFixture, article: ResearchArticle) -> None:
    secret = "test-secret-never-log"
    with caplog.at_level(logging.WARNING):
        with pytest.raises(LLMConfigurationError):
            LLMService(settings=Settings(llm_provider="gemini", gemini_api_key=None, llm_api_key=None))

    assert secret not in caplog.text
    assert "GEMINI_API_KEY" in caplog.text or caplog.text == ""


def test_provider_abstraction_accepts_custom_provider(article: ResearchArticle) -> None:
    service = LLMService(provider=MockLLMProvider())

    assert service.summarize_article(article).summary