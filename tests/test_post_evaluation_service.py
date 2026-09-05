"""Offline tests for combined Phase 7 post evaluation."""

from datetime import datetime, timezone

from services.fact_checker_service import FactCheckStatus, FactCheckerService
from services.linkedin_post_service import LinkedInPost
from services.post_evaluation_service import EvaluationStatus, PostEvaluationService
from services.quality_checker_service import QualityStatus
from services.research_service import ResearchArticle


def article(summary: str = "The model combines language and vision capabilities.") -> ResearchArticle:
    return ResearchArticle(
        title="A multimodal AI model",
        summary=summary,
        source="Example AI Research",
        source_url="https://example.com/article",
        category="Multimodal AI",
        published_at=datetime.now(timezone.utc),
    )


def post(body: str = "The model combines language and vision capabilities.") -> LinkedInPost:
    return LinkedInPost(
        title="A practical AI development",
        hook="AI progress matters when it changes practical work.",
        body=body,
        key_insights=["The model combines language and vision capabilities."],
        practical_impact="Teams can evaluate this development against practical needs.",
        engagement_question="Which AI workflow would benefit from this development?",
        hashtags=["#MultimodalAI", "#AI", "#ComputerVision"],
        source_url="https://example.com/article",
        source_name="Example AI Research",
        topic_category="Multimodal AI",
        content=(
            "AI progress matters when it changes practical work.\n\n"
            + body
            + "\n\nTeams can evaluate this development against practical needs.\n\n"
            "Which AI workflow would benefit from this development?\n\n"
            "#MultimodalAI #AI #ComputerVision"
        ),
    )


def test_good_post_is_approved() -> None:
    result = PostEvaluationService().evaluate_post(post(), article())

    assert result.final_status == EvaluationStatus.APPROVED
    assert result.fact_check.overall_status == FactCheckStatus.PASS


def test_minor_quality_issue_needs_revision() -> None:
    draft = post().model_copy(update={"content": post().content + "!!!!"})

    result = PostEvaluationService().evaluate_post(draft, article())

    assert result.final_status in {EvaluationStatus.NEEDS_REVISION, EvaluationStatus.REJECTED}
    assert result.quality_check.issues


def test_unsupported_claim_needs_revision_or_rejection() -> None:
    draft = post("The model achieved 99% accuracy.")

    result = PostEvaluationService().evaluate_post(draft, article())

    assert result.final_status in {EvaluationStatus.NEEDS_REVISION, EvaluationStatus.REJECTED}
    assert result.fact_check.unsupported_claims >= 1


def test_serious_hallucination_is_rejected() -> None:
    draft = post("The model achieved 99% accuracy. It has 10 billion users.")

    result = PostEvaluationService().evaluate_post(draft, article())

    assert result.final_status == EvaluationStatus.REJECTED
    assert result.blocking_issues


def test_secret_is_rejected() -> None:
    draft = post().model_copy(update={"content": "GEMINI_API_KEY=secret-value"})

    result = PostEvaluationService().evaluate_post(draft, article())

    assert result.final_status == EvaluationStatus.REJECTED
    assert result.quality_check.status == QualityStatus.FAIL


def test_empty_post_is_rejected() -> None:
    draft = post().model_copy(update={"content": ""})

    result = PostEvaluationService().evaluate_post(draft, article())

    assert result.final_status == EvaluationStatus.REJECTED


def test_missing_source_is_rejected() -> None:
    source = ResearchArticle.model_construct(
        title="", summary="", source="", source_url="", category="General AI"
    )

    result = PostEvaluationService().evaluate_post(post(), source)

    assert result.final_status == EvaluationStatus.REJECTED


def test_scores_are_combined_and_bounded() -> None:
    result = PostEvaluationService().evaluate_post(post(), article())

    assert 0 <= result.overall_score <= 10
    assert result.overall_score == round(
        result.fact_check.overall_score * 0.55
        + result.quality_check.overall_score * 0.45,
        2,
    )


def test_deterministic_security_gate_cannot_be_overridden() -> None:
    result = PostEvaluationService().evaluate_post(
        post().model_copy(update={"content": "api_key=secret"}), article()
    )

    assert result.final_status == EvaluationStatus.REJECTED


def test_evaluation_does_not_modify_draft() -> None:
    draft = post()
    original = draft.model_copy(deep=True)

    PostEvaluationService().evaluate_post(draft, article())

    assert draft == original


def test_evaluation_has_no_publishing_behavior() -> None:
    assert not hasattr(PostEvaluationService(), "publish_post")