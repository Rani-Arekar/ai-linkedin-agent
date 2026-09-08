"""Structured, draft-only LinkedIn post generation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database import crud
from database.models import Post, Topic
from services.llm_service import LLMService, LLMServiceError, load_prompt
from services.topic_selection_service import TopicCandidate, TopicSelectionResult

logger = logging.getLogger(__name__)


class PostGenerationError(RuntimeError):
    """Raised when a LinkedIn draft cannot be generated or validated."""


class PostMetrics(BaseModel):
    """Descriptive metrics calculated from draft text."""

    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    paragraph_count: int = Field(ge=0)
    hashtag_count: int = Field(ge=0)
    has_engagement_question: bool
    has_hook: bool


class LinkedInPost(BaseModel):
    """Structured LinkedIn draft with source attribution and computed metrics."""

    title: str = Field(min_length=1, max_length=200)
    hook: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=8000)
    key_insights: list[str] = Field(min_length=1, max_length=6)
    practical_impact: str = Field(min_length=1, max_length=2000)
    engagement_question: str = Field(min_length=1, max_length=500)
    hashtags: list[str] = Field(min_length=1, max_length=10)

    source_url: str = Field(min_length=1, max_length=2048)
    source_name: str = Field(min_length=1, max_length=255)
    topic_category: str = Field(min_length=1, max_length=100)

    content: str = ""
    character_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    status: str = Field(
        default="draft",
        min_length=1,
        max_length=50,
    )

    @field_validator("hashtags")
    @classmethod
    def validate_hashtags(cls, values: list[str]) -> list[str]:
        """Validate hashtag syntax and uniqueness."""

        if not values:
            raise ValueError("at least one hashtag is required")

        cleaned: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = value.strip()

            if not re.fullmatch(r"#[A-Za-z][A-Za-z0-9_]*", value):
                raise ValueError(
                    "hashtags must begin with # and contain no spaces"
                )

            normalized = value.lower()

            if normalized in seen:
                raise ValueError("hashtags must not be duplicated")

            seen.add(normalized)
            cleaned.append(value)

        return cleaned


@dataclass(frozen=True, slots=True)
class _TopicData:
    """Normalized topic fields used to construct an LLM prompt."""

    title: str
    summary: str
    source_url: str
    source_name: str
    category: str
    analysis: Any = None


class LinkedInPostService:
    """Generate and validate human-reviewable LinkedIn drafts without publishing."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_service = llm_service or LLMService(settings=self.settings)

    def generate_post(
        self,
        topic: TopicCandidate | TopicSelectionResult,
        revision_feedback: Sequence[str] | None = None,
    ) -> LinkedInPost:
        """Generate one structured draft from a selected Phase 5 topic."""

        topic_data = self._topic_data(topic)

        if not topic_data.title.strip() or not topic_data.summary.strip():
            raise ValueError("Selected topic must contain a title and summary")

        prompt = load_prompt("linkedin_post_prompt.txt")

        prompt += "\n\nTOPIC:\n"
        prompt += json.dumps(
            {
                "title": topic_data.title,
                "summary": topic_data.summary,
                "source_url": topic_data.source_url,
                "source_name": topic_data.source_name,
                "category": topic_data.category,
                "analysis": topic_data.analysis,
            },
            default=lambda value: value.model_dump(mode="json")
            if hasattr(value, "model_dump")
            else str(value),
            ensure_ascii=True,
        )

        if revision_feedback:
            prompt += "\n\nREVISION FEEDBACK TO ADDRESS:\n"
            prompt += "\n".join(f"- {item}" for item in revision_feedback)

        try:
            result = self.llm_service.generate_structured(
                prompt,
                LinkedInPost,
            )
        except LLMServiceError as exc:
            logger.warning(
                "LinkedIn draft generation failed: %s",
                type(exc).__name__,
            )
            raise PostGenerationError(
                "Unable to generate LinkedIn draft"
            ) from exc
        except Exception as exc:
            logger.warning(
                "Unexpected LinkedIn draft failure: %s",
                type(exc).__name__,
            )
            raise PostGenerationError(
                "Unable to generate LinkedIn draft"
            ) from exc

        if not isinstance(result, LinkedInPost):
            raise PostGenerationError(
                "LLM returned an invalid LinkedIn draft"
            )

        post = result.model_copy(
            update={
                "hashtags": self.generate_hashtags(
                    topic_data.category,
                    topic_data.title,
                ),
                "source_url": topic_data.source_url,
                "source_name": topic_data.source_name,
                "topic_category": topic_data.category,
                "status": "draft",
            }
        )

        post.content = self.format_post(post)
        metrics = self.calculate_post_metrics(post.content)

        post.character_count = metrics.character_count
        post.word_count = metrics.word_count

        self.validate_post(post, metrics)

        return post

    def validate_post(
        self,
        post: LinkedInPost,
        metrics: PostMetrics | None = None,
    ) -> bool:
        """Validate content safety, hashtags, structure, repetition, and configured length."""

        content = post.content.strip()

        if not content:
            raise PostGenerationError("LinkedIn draft is empty")

        if metrics is None:
            metrics = self.calculate_post_metrics(content)
        self._validate_content_safety(content)
        if (
            metrics.character_count
            < self.settings.linkedin_post_min_characters
        ):
            raise PostGenerationError(
                "LinkedIn draft is shorter than the configured minimum"
            )

        if (
            metrics.character_count
            > self.settings.linkedin_post_max_characters
        ):
            raise PostGenerationError(
                "LinkedIn draft exceeds the configured maximum length"
            )

        if not post.hook.strip() or not post.engagement_question.strip():
            raise PostGenerationError(
                "Draft requires a hook and a question-ending engagement prompt"
            )

        if not post.engagement_question.strip().endswith("?"):
            raise PostGenerationError(
                "Draft requires a hook and a question-ending engagement prompt"
            )

        normalized_hashtags = [item.lower() for item in post.hashtags]

        if len(normalized_hashtags) != len(set(normalized_hashtags)):
            raise PostGenerationError("Draft contains duplicate hashtags")

        if post.status != "draft":
            raise PostGenerationError(
                "Generated posts must remain drafts"
            )

        return True

    @staticmethod
    def generate_hashtags(category: str, title: str) -> list[str]:
        """Return 3-5 relevant, category-dependent hashtags."""

        hashtag_map = {
            "Generative AI": (
                "#GenerativeAI",
                "#ArtificialIntelligence",
                "#AI",
            ),
            "Large Language Models": (
                "#LLM",
                "#GenerativeAI",
                "#ArtificialIntelligence",
            ),
            "Computer Vision": (
                "#ComputerVision",
                "#DeepLearning",
                "#AI",
            ),
            "Multimodal AI": (
                "#MultimodalAI",
                "#ComputerVision",
                "#AI",
            ),
            "AI Agents": (
                "#AIAgents",
                "#ArtificialIntelligence",
                "#Automation",
            ),
            "Robotics": (
                "#Robotics",
                "#ArtificialIntelligence",
                "#Automation",
            ),
            "MLOps": (
                "#MLOps",
                "#MachineLearning",
                "#AI",
            ),
            "Responsible AI": (
                "#ResponsibleAI",
                "#AISafety",
                "#ArtificialIntelligence",
            ),
            "Deep Learning": (
                "#DeepLearning",
                "#NeuralNetworks",
                "#AI",
            ),
            "Machine Learning": (
                "#MachineLearning",
                "#DataScience",
                "#AI",
            ),
            "NLP": (
                "#NLP",
                "#LanguageModels",
                "#AI",
            ),
            "AI Tools": (
                "#AITools",
                "#DeveloperTools",
                "#AI",
            ),
        }

        hashtags = list(
            hashtag_map.get(
                category,
                (
                    "#ArtificialIntelligence",
                    "#MachineLearning",
                    "#AI",
                ),
            )
        )

        title_lower = title.lower()

        if "transformer" in title_lower:
            hashtags.append("#Transformers")

        # Remove duplicates while preserving order.
        result: list[str] = []
        seen: set[str] = set()

        for hashtag in hashtags:
            normalized = hashtag.lower()
            if normalized not in seen:
                seen.add(normalized)
                result.append(hashtag)

        max_hashtags = getattr(
            get_settings(),
            "linkedin_max_hashtags",
            5,
        )

        return result[:max_hashtags]

    @staticmethod
    def calculate_post_metrics(content: str) -> PostMetrics:
        """Calculate descriptive text metrics without predicting engagement."""

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", content.strip())
            if paragraph.strip()
        ]

        words = re.findall(
            r"\b\w+[\w'-]*\b",
            content,
        )

        hashtags = re.findall(
            r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*",
            content,
        )

        return PostMetrics(
            character_count=len(content),
            word_count=len(words),
            paragraph_count=len(paragraphs),
            hashtag_count=len(hashtags),
            has_engagement_question=bool(
                re.search(r"\?\s*(?:\n|$)", content)
            ),
            has_hook=bool(paragraphs),
        )

    @staticmethod
    def format_post(post: LinkedInPost) -> str:
        """Format structured sections into readable draft text."""

        insights = "\n".join(
            f"- {item.strip()}"
            for item in post.key_insights
            if item.strip()
        )

        parts = [
            post.hook.strip(),
            post.body.strip(),
            "\n\nKey insights:\n" + insights,
            "\n\nPractical impact:\n" + post.practical_impact.strip(),
            post.engagement_question.strip(),
            " ".join(post.hashtags),
        ]

        return "\n\n".join(
            part.strip()
            for part in parts
            if part.strip()
        )

    def save_draft(
    self,
    session: Session,
    post: LinkedInPost,
    topic: TopicCandidate,
    quality_score: float | None = None,
    fact_check_status: str | None = None,
    similarity_score: float | None = None,
) -> Post:
        """Store a generated draft with evaluation results; never publish it."""

        database_topic = session.scalar(
            select(Topic).where(Topic.source_url == topic.url)
        )

        if database_topic is None:
            raise ValueError(
                "Selected topic must already exist in the database"
            )

        saved_post = crud.create_post(
            session,
            topic_id=database_topic.id,
            content=post.content,
            status="draft",
        )

        # Store evaluation results when the database model supports them.
        if quality_score is not None:
            saved_post.quality_score = quality_score

        if fact_check_status is not None:
            saved_post.fact_check_status = fact_check_status

        if similarity_score is not None:
            saved_post.similarity_score = similarity_score

        session.flush()

        return saved_post

    @staticmethod
    def _topic_data(
        topic: TopicCandidate | TopicSelectionResult,
    ) -> _TopicData:
        """Normalize a topic or topic-selection result."""

        if isinstance(topic, TopicSelectionResult):
            topic = topic.selected_topic

        if topic is None:
            raise ValueError("A selected topic is required")

        return _TopicData(
            title=topic.title,
            summary=topic.summary,
            source_url=topic.url,
            source_name=topic.source,
            category=topic.category,
            analysis=getattr(topic, "llm_analysis", None),
        )

    @staticmethod
    def _validate_content_safety(content: str) -> None:
        """Reject obvious secrets, placeholders, and excessive repetition."""

        lowered = content.lower()

        secret_markers = (
            "AIza",
            "sk-",
            "gemini_api_key",
            "linkedin_client_secret",
            "begin private key",
        )

        if any(marker.lower() in lowered for marker in secret_markers):
            raise PostGenerationError(
                "Draft contains a possible secret"
            )

        placeholders = (
            "lorem ipsum",
            "[your text",
            "todo:",
        )

        if any(marker in lowered for marker in placeholders):
            raise PostGenerationError(
                "Draft contains placeholder text"
            )

        words = re.findall(r"\b\w+\b", lowered)

        if len(words) >= 30:
            counts: dict[str, int] = {}

            for word in words:
                counts[word] = counts.get(word, 0) + 1

            most_common = max(counts.values())

            if most_common / len(words) > 0.35:
                raise PostGenerationError(
                    "Draft is excessively repetitive"
                )