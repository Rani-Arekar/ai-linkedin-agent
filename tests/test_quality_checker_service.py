"""Offline tests for deterministic quality and safety checks."""

from datetime import datetime, timezone

from config import Settings
from services.linkedin_post_service import LinkedInPost
from services.quality_checker_service import QualityCheckerService, QualityStatus


def make_post(content: str | None = None, hashtags: list[str] | None = None) -> LinkedInPost:
    content = (
        content
        if content is not None
        else (
        "AI systems become more useful when technical progress connects to practical work.\n\n"
        "This article describes a model with language and vision capabilities.\n\n"
        "The main lesson is to evaluate new capabilities against real workflows.\n\n"
        "Which AI workflow would benefit most from this development?\n\n"
        "#MultimodalAI #AI #ComputerVision"
        )
    )
    return LinkedInPost(
        title="A practical AI development",
        hook="AI systems become more useful when technical progress connects to practical work.",
        body="This article describes a model with language and vision capabilities.",
        key_insights=["Evaluate new capabilities against real workflows."],
        practical_impact="Teams can assess where this capability creates practical value.",
        engagement_question="Which AI workflow would benefit most from this development?",
        hashtags=hashtags or ["#MultimodalAI", "#AI", "#ComputerVision"],
        source_url="https://example.com/article",
        source_name="Example AI Research",
        topic_category="Multimodal AI",
        content=content,
    )


def test_service_initialization() -> None:
    assert isinstance(QualityCheckerService(), QualityCheckerService)


def test_good_post_passes() -> None:
    result = QualityCheckerService().check_post(make_post())

    assert result.status == QualityStatus.PASS
    assert result.overall_score >= 7


def test_poor_readability_is_flagged() -> None:
    content = "A very long sentence " * 40 + "?"
    result = QualityCheckerService().check_post(make_post(content))

    assert "Very long sentences reduce readability." in result.issues


def test_excessive_emojis_are_flagged() -> None:
    result = QualityCheckerService().check_post(make_post("AI development matters. " + "🚀" * 5 + "\n\nWhat changes next?"))

    assert "Excessive emoji usage." in result.issues


def test_excessive_punctuation_is_flagged() -> None:
    result = QualityCheckerService().check_post(make_post("This changes everything!!!!!!"))

    assert "Excessive punctuation." in result.issues


def test_excessive_hashtags_are_flagged() -> None:
    result = QualityCheckerService().check_post(
        make_post(hashtags=["#AI", "#ML", "#NLP", "#CV", "#LLM", "#Robotics"])
    )

    assert "Too many hashtags." in result.issues


def test_duplicate_hashtags_are_flagged() -> None:
    post = make_post()
    post = post.model_construct(**{**post.model_dump(), "hashtags": ["#AI", "#ai", "#ML"]})
    result = QualityCheckerService().check_post(post)

    assert "Duplicate hashtags detected." in result.issues


def test_clickbait_is_flagged() -> None:
    result = QualityCheckerService().check_post(make_post("YOU WON'T BELIEVE this AI update!"))

    assert "Obvious clickbait language detected." in result.issues


def test_secret_leakage_fails() -> None:
    result = QualityCheckerService().check_post(make_post("GEMINI_API_KEY=secret-value"))

    assert result.status == QualityStatus.FAIL
    assert any("credential" in issue for issue in result.issues)
    assert "secret-value" not in str(result.model_dump())


def test_placeholder_is_flagged() -> None:
    result = QualityCheckerService().check_post(make_post("lorem ipsum " * 20))

    assert result.status in {QualityStatus.NEEDS_REVIEW, QualityStatus.FAIL}


def test_missing_engagement_question_is_flagged() -> None:
    post = make_post()
    post = post.model_copy(update={"engagement_question": "A useful takeaway."})
    result = QualityCheckerService().check_post(post)

    assert "Missing meaningful engagement question." in result.issues


def test_repetitive_content_is_flagged() -> None:
    result = QualityCheckerService().check_post(make_post("AI " * 100))

    assert "Repetitive text detected." in result.issues


def test_empty_post_fails() -> None:
    result = QualityCheckerService().check_post(make_post(""))

    assert result.status == QualityStatus.FAIL
    assert result.overall_score == 0


def test_scores_are_bounded_and_deterministic() -> None:
    service = QualityCheckerService(Settings())
    first = service.check_post(make_post())
    second = service.check_post(make_post())

    assert first == second
    for field in ("overall_score", "readability_score", "spam_score", "clickbait_score"):
        assert 0 <= getattr(first, field) <= 10