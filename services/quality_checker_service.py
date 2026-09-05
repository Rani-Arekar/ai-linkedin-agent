"""Deterministic readability, professionalism, spam, and security checks."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field

from config import Settings, get_settings
from services.linkedin_post_service import LinkedInPost


class QualityStatus(StrEnum):
    """Overall quality decision."""

    PASS = "PASS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAIL = "FAIL"


class QualityCheckResult(BaseModel):
    """Bounded deterministic quality evaluation."""

    overall_score: float = Field(ge=0, le=10)
    readability_score: float = Field(ge=0, le=10)
    professionalism_score: float = Field(ge=0, le=10)
    structure_score: float = Field(ge=0, le=10)
    engagement_score: float = Field(ge=0, le=10)
    hashtag_score: float = Field(ge=0, le=10)
    spam_score: float = Field(ge=0, le=10)
    clickbait_score: float = Field(ge=0, le=10)
    issues: list[str] = Field(default_factory=list, max_length=20)
    recommendations: list[str] = Field(default_factory=list, max_length=20)
    status: QualityStatus


class QualityCheckerService:
    """Evaluate a draft using repeatable checks; no engagement claims are made."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def check_post(self, post: LinkedInPost) -> QualityCheckResult:
        """Return deterministic quality dimensions and actionable issues."""

        content = post.content.strip()
        issues: list[str] = []
        recommendations: list[str] = []
        if not content:
            return self._failed("The post is empty.", "Generate substantive draft content.")
        metrics = self._metrics(content)
        security_issue = self._security_issue(content)
        if security_issue:
            issues.append(security_issue)
            recommendations.append("Remove credential-like content before review.")
        if metrics["character_count"] < self.settings.linkedin_post_min_characters or metrics["character_count"] > self.settings.linkedin_post_max_characters:
            issues.append("Post length is outside the configured range.")
            recommendations.append("Adjust the draft length without losing factual context.")
        if metrics["emoji_count"] > 3:
            issues.append("Excessive emoji usage.")
            recommendations.append("Use fewer emojis in this technical draft.")
        if metrics["punctuation_count"] > 4:
            issues.append("Excessive punctuation.")
        if metrics["uppercase_ratio"] > 0.35:
            issues.append("Excessive capitalization.")
        if any(pattern in content.lower() for pattern in ("you won't believe", "shocking!", "changes everything!!!", "secret trick", "nobody is talking about this")):
            issues.append("Obvious clickbait language detected.")
            recommendations.append("Replace the clickbait hook with a specific evidence-based hook.")
        if len(post.hashtags) > self.settings.linkedin_max_hashtags:
            issues.append("Too many hashtags.")
        if len(set(tag.lower() for tag in post.hashtags)) != len(post.hashtags):
            issues.append("Duplicate hashtags detected.")
        if not post.engagement_question.strip().endswith("?"):
            issues.append("Missing meaningful engagement question.")
        if metrics["repetition_ratio"] < 0.42:
            issues.append("Repetitive text detected.")
        if metrics["average_sentence_words"] > 28:
            issues.append("Very long sentences reduce readability.")
            recommendations.append("Break long sentences into shorter paragraphs.")
        readability = max(0.0, 10.0 - (2 if metrics["average_sentence_words"] > 28 else 0) - (2 if metrics["paragraph_count"] < 3 else 0))
        professionalism = max(0.0, 10.0 - len([item for item in issues if item not in {"Post length is outside the configured range."}]))
        structure = 10.0 if metrics["paragraph_count"] >= 4 and metrics["has_question"] else 7.0
        engagement = 9.0 if metrics["has_question"] else 4.0
        hashtag_score = 10.0 if 3 <= len(post.hashtags) <= self.settings.linkedin_max_hashtags else 5.0
        spam_score = max(0.0, 10.0 - sum(issue in issues for issue in ("Excessive emoji usage.", "Excessive punctuation.", "Repetitive text detected.", "Too many hashtags.")) * 2)
        clickbait_score = 3.0 if "Obvious clickbait language detected." in issues else 10.0
        overall = round(sum((readability, professionalism, structure, engagement, hashtag_score, spam_score, clickbait_score)) / 7, 2)
        if security_issue:
            status = QualityStatus.FAIL
        elif len(issues) >= 3 or overall < 5:
            status = QualityStatus.FAIL
        elif issues or overall < 7:
            status = QualityStatus.NEEDS_REVIEW
        else:
            status = QualityStatus.PASS
        return QualityCheckResult(
            overall_score=overall,
            readability_score=readability,
            professionalism_score=professionalism,
            structure_score=structure,
            engagement_score=engagement,
            hashtag_score=hashtag_score,
            spam_score=spam_score,
            clickbait_score=clickbait_score,
            issues=issues,
            recommendations=recommendations,
            status=status,
        )

    @staticmethod
    def _metrics(content: str) -> dict[str, float | int | bool]:
        sentences = [item for item in re.split(r"[.!?]+", content) if item.strip()]
        words = re.findall(r"\b\w+\b", content)
        unique = len(set(word.lower() for word in words))
        letters = [char for char in content if char.isalpha()]
        return {
            "character_count": len(content),
            "paragraph_count": len([item for item in content.split("\n\n") if item.strip()]),
            "emoji_count": len(re.findall(r"[^\x00-\x7F]", content)),
            "punctuation_count": len(re.findall(r"[!?]", content)) + len(re.findall(r"\.{3,}", content)) * 3,
            "uppercase_ratio": sum(char.isupper() for char in letters) / max(len(letters), 1),
            "average_sentence_words": len(words) / max(len(sentences), 1),
            "repetition_ratio": unique / max(len(words), 1),
            "has_question": bool(re.search(r"\?\s*(?:\n|$)", content)),
        }

    @staticmethod
    def _security_issue(content: str) -> str | None:
        markers = ("AIza", "sk-", "api_key=", "password=", "authorization: bearer", "begin private key", "gemini_api_key", "linkedin_client_secret")
        return "Possible credential or secret leakage detected." if any(marker.lower() in content.lower() for marker in markers) else None

    @staticmethod
    def _failed(issue: str, recommendation: str) -> QualityCheckResult:
        return QualityCheckResult(
            overall_score=0,
            readability_score=0,
            professionalism_score=0,
            structure_score=0,
            engagement_score=0,
            hashtag_score=0,
            spam_score=0,
            clickbait_score=0,
            issues=[issue],
            recommendations=[recommendation],
            status=QualityStatus.FAIL,
        )