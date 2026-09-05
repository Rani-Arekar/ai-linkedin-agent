"""Offline tests for source-grounded fact checking."""

from datetime import datetime, timezone

import pytest

from services.fact_checker_service import (
    ClaimStatus,
    FactCheckStatus,
    FactCheckerService,
)
from services.linkedin_post_service import LinkedInPost
from services.llm_service import LLMProviderError, LLMService, MockLLMProvider
from services.research_service import ResearchArticle


def make_article(summary: str = "The model combines language and vision capabilities.") -> ResearchArticle:
    return ResearchArticle(
        title="OpenAI releases a multimodal AI model",
        summary=summary,
        source="Example AI Research",
        source_url="https://example.com/article",
        category="Multimodal AI",
        published_at=datetime.now(timezone.utc),
    )


def make_post(body: str = "The model combines language and vision capabilities.") -> LinkedInPost:
    return LinkedInPost(
        title="A practical AI development",
        hook="AI progress matters when it changes what teams can build.",
        body=body,
        key_insights=["The model combines language and vision capabilities."],
        practical_impact="Teams can evaluate this development against practical needs.",
        engagement_question="Which AI workflow would benefit from this development?",
        hashtags=["#MultimodalAI", "#AI", "#ArtificialIntelligence"],
        source_url="https://example.com/article",
        source_name="Example AI Research",
        topic_category="Multimodal AI",
        content=body,
    )


def test_service_initialization() -> None:
    assert isinstance(FactCheckerService(), FactCheckerService)


def test_supported_claim() -> None:
    result = FactCheckerService().check_post(make_post(), make_article())

    assert result.overall_status == FactCheckStatus.PASS
    assert result.supported_claims >= 1
    assert result.claims[0].status == ClaimStatus.SUPPORTED


def test_unsupported_claim() -> None:
    result = FactCheckerService().check_post(
        make_post("The model achieved a 99% benchmark score."), make_article()
    )

    assert result.unsupported_claims >= 1
    assert result.claims[0].status == ClaimStatus.UNSUPPORTED
    assert result.overall_status == FactCheckStatus.NEEDS_REVIEW


def test_partially_supported_claim() -> None:
    result = FactCheckerService().check_post(
        make_post("The model combines language capabilities for enterprise deployment."),
        make_article(),
    )

    assert result.claims[0].status == ClaimStatus.PARTIALLY_SUPPORTED


def test_uncertain_claim() -> None:
    result = FactCheckerService().check_post(
        make_post("This development will transform every industry worldwide."),
        make_article(),
    )

    assert result.claims[0].status == ClaimStatus.UNCERTAIN
    assert result.uncertain_claims == 1


def test_multiple_claims_are_extracted() -> None:
    post = make_post(
        "The model combines language and vision capabilities. "
        "The release happened in 2024."
    )
    result = FactCheckerService().check_post(post, make_article())

    assert len(result.claims) == 2
    assert result.claims[1].status == ClaimStatus.UNSUPPORTED


def test_numerical_claim_mismatch_is_not_supported() -> None:
    result = FactCheckerService().check_post(
        make_post("The model is 40% faster than previous systems."),
        make_article("The model is 30% faster than previous systems."),
    )

    assert result.claims[0].status == ClaimStatus.UNSUPPORTED
    assert result.hallucination_risk > 0


def test_date_mismatch_is_not_supported() -> None:
    result = FactCheckerService().check_post(
        make_post("OpenAI released the model in 2025."),
        make_article("OpenAI released the model in 2024."),
    )

    assert result.claims[0].status == ClaimStatus.UNSUPPORTED


def test_company_claim_requires_source_evidence() -> None:
    result = FactCheckerService().check_post(
        make_post("Microsoft launched this product for all customers."), make_article()
    )

    assert result.claims[0].status in {ClaimStatus.UNSUPPORTED, ClaimStatus.UNCERTAIN}


def test_empty_source_fails() -> None:
    result = FactCheckerService().check_post(make_post(), make_article(""))
    result = FactCheckerService().check_post(
        make_post(), ResearchArticle.model_construct(
            title="", summary="", source="", source_url="", category="General AI"
        )
    )

    assert result.overall_status == FactCheckStatus.FAIL
    assert result.overall_score == 0


def test_empty_post_fails() -> None:
    post = make_post()
    post = post.model_copy(update={"content": ""})

    result = FactCheckerService().check_post(post, make_article())

    assert result.overall_status == FactCheckStatus.FAIL


def test_claim_extraction_excludes_questions() -> None:
    post = make_post()
    post = post.model_copy(update={"engagement_question": "What will change?"})

    claims = FactCheckerService().extract_claims(post)

    assert all(not claim.endswith("?") for claim in claims)


def test_llm_evidence_enrichment_uses_existing_service() -> None:
    checker = FactCheckerService(LLMService(provider=MockLLMProvider()))

    result = checker.check_post(make_post(), make_article())

    assert result.claims


def test_llm_failure_does_not_crash_fact_check() -> None:
    class FailingLLM:
        def analyze_article(self, article: ResearchArticle) -> object:
            raise LLMProviderError("provider failure")

    result = FactCheckerService(FailingLLM()).check_post(make_post(), make_article())

    assert result.overall_status in {FactCheckStatus.PASS, FactCheckStatus.NEEDS_REVIEW}


def test_fact_scores_and_risk_are_bounded() -> None:
    result = FactCheckerService().check_post(
        make_post("A 99% faster model launched in 2025."), make_article()
    )

    assert 0 <= result.overall_score <= 10
    assert 0 <= result.hallucination_risk <= 1


def test_fact_check_is_deterministic() -> None:
    service = FactCheckerService()
    first = service.check_post(make_post(), make_article())
    second = service.check_post(make_post(), make_article())

    assert first == second