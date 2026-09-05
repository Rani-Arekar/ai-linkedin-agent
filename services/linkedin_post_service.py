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
    body: str = Field(min_length=1, max_length=8_000)
    key_insights: list[str] = Field(min_length=1, max_length=6)
    practical_impact: str = Field(min_length=1, max_length=2_000)
    engagement_question: str = Field(min_length=1, max_length=500)
    hashtags: list[str] = Field(min_length=1, max_length=10)
    source_url: str = Field(min_length=1, max_length=2048)
    source_name: str = Field(min_length=1, max_length=255)
    topic_category: str = Field(min_length=1, max_length=100)
    content: str = ""
    character_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    status: str = Field(default="draft", min_length=1, max_length=50)

    @field_validator("hashtags")
    @classmethod
    def validate_hashtags(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            tag = value.strip()
            if not re.fullmatch(r"#[A-Za-z][A-Za-z0-9_]*", tag):
                raise ValueError("hashtags must begin with # and contain no spaces")
            if tag.lower() not in {item.lower() for item in normalized}:
                normalized.append(tag)
        if not normalized:
            raise ValueError("at least one hashtag is required")
        if len(normalized) != len(values):
            raise ValueError("hashtags must not be duplicated")
        return normalized


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
        prompt = load_prompt("linkedin_post_prompt.txt") + "\n\nTOPIC:\n" + json.dumps(
            {
                "title": topic_data.title,
                "summary": topic_data.summary,
                "source_url": topic_data.source_url,
                "source_name": topic_data.source_name,
                "category": topic_data.category,
                "analysis": topic_data.analysis,
            },
            default=lambda value: value.model_dump(mode="json"),
            ensure_ascii=True,
        )
        if revision_feedback:
            prompt += "\n\nREVISION FEEDBACK TO ADDRESS:\n" + "\n".join(
                f"- {item}" for item in revision_feedback
            )
        try:
            draft = self.llm_service.generate_structured(prompt, LinkedInPost)
        except LLMServiceError as error:
            logger.warning("LinkedIn draft generation failed: %s", type(error).__name__)
            raise PostGenerationError("Unable to generate LinkedIn draft") from error
        except Exception as error:
            logger.warning("Unexpected LinkedIn draft failure: %s", type(error).__name__)
            raise PostGenerationError("Unable to generate LinkedIn draft") from error
        if not isinstance(draft, LinkedInPost):
            raise PostGenerationError("LLM returned an invalid LinkedIn draft")
        draft = draft.model_copy(
            update={
                "hashtags": self.generate_hashtags(topic_data.category, topic_data.title),
                "source_url": topic_data.source_url,
                "source_name": topic_data.source_name,
                "topic_category": topic_data.category,
                "status": "draft",
            }
        )
        content = self.format_post(draft)
        metrics = self.calculate_post_metrics(content)
        draft = draft.model_copy(
            update={
                "content": content,
                "character_count": metrics.character_count,
                "word_count": metrics.word_count,
            }
        )
        self.validate_post(draft, metrics)
        return draft

    def validate_post(
        self, post: LinkedInPost, metrics: PostMetrics | None = None
    ) -> bool:
        """Validate content safety, hashtags, structure, repetition, and configured length."""

        if not post.content.strip():
            raise PostGenerationError("LinkedIn draft is empty")
        metrics = metrics or self.calculate_post_metrics(post.content)
        self._validate_content_safety(post.content)
        if metrics.character_count < self.settings.linkedin_post_min_characters:
            raise PostGenerationError("LinkedIn draft is shorter than the configured minimum")
        if metrics.character_count > self.settings.linkedin_post_max_characters:
            raise PostGenerationError("LinkedIn draft exceeds the configured maximum length")
        if not post.hook.strip() or not post.engagement_question.strip().endswith("?"):
            raise PostGenerationError("Draft requires a hook and a question-ending engagement prompt")
        if len(set(tag.lower() for tag in post.hashtags)) != len(post.hashtags):
            raise PostGenerationError("Draft contains duplicate hashtags")
        if post.status != "draft":
            raise PostGenerationError("Generated posts must remain drafts")
        return True

    def generate_hashtags(self, category: str, title: str = "") -> list[str]:
        """Return 3-5 relevant, category-dependent hashtags."""

        mapping = {
            "Generative AI": ["#GenerativeAI", "#ArtificialIntelligence", "#AI"],
            "Large Language Models": ["#LLM", "#GenerativeAI", "#ArtificialIntelligence"],
            "Computer Vision": ["#ComputerVision", "#DeepLearning", "#AI"],
            "Multimodal AI": ["#MultimodalAI", "#ComputerVision", "#AI"],
            "AI Agents": ["#AIAgents", "#ArtificialIntelligence", "#Automation"],
            "Robotics": ["#Robotics", "#ArtificialIntelligence", "#Automation"],
            "MLOps": ["#MLOps", "#MachineLearning", "#AI"],
            "Responsible AI": ["#ResponsibleAI", "#AISafety", "#ArtificialIntelligence"],
            "Deep Learning": ["#DeepLearning", "#NeuralNetworks", "#AI"],
            "Machine Learning": ["#MachineLearning", "#DataScience", "#AI"],
            "NLP": ["#NLP", "#LanguageModels", "#AI"],
            "AI Tools": ["#AITools", "#DeveloperTools", "#AI"],
        }
        tags = list(mapping.get(category, ["#ArtificialIntelligence", "#MachineLearning", "#AI"]))
        lowered_title = title.lower()
        if "transformer" in lowered_title and "#Transformers" not in tags:
            tags.append("#Transformers")
        return tags[: self.settings.linkedin_max_hashtags]

    def calculate_post_metrics(self, content: str) -> PostMetrics:
        """Calculate descriptive text metrics without predicting engagement."""

        paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", content.strip()) if paragraph.strip()]
        return PostMetrics(
            character_count=len(content),
            word_count=len(re.findall(r"\b\w+[\w'-]*\b", content)),
            paragraph_count=len(paragraphs),
            hashtag_count=len(re.findall(r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*", content)),
            has_engagement_question=bool(re.search(r"\?\s*(?:\n|$)", content.strip())),
            has_hook=bool(paragraphs),
        )

    def format_post(self, post: LinkedInPost) -> str:
        """Format structured sections into readable draft text."""

        insights = "\n".join(f"- {insight.strip()}" for insight in post.key_insights)
        hashtags = " ".join(post.hashtags)
        return (
            f"{post.hook.strip()}\n\n{post.body.strip()}\n\n"
            f"Key insights:\n{insights}\n\n"
            f"Practical impact:\n{post.practical_impact.strip()}\n\n"
            f"{post.engagement_question.strip()}\n\n{hashtags}"
        )

    def save_draft(
        self, session: Session, post: LinkedInPost, topic: TopicCandidate
    ) -> Post:
        """Store a generated draft against an existing Topic; never publish it."""

        database_topic = session.scalar(
            select(Topic).where(Topic.source_url == topic.url)
        )
        if database_topic is None:
            raise ValueError("Selected topic must already exist in the database")
        return crud.create_post(
            session,
            topic_id=database_topic.id,
            content=post.content,
            status="draft",
        )

    @staticmethod
    def _topic_data(topic: TopicCandidate | TopicSelectionResult) -> _TopicData:
        selected = topic.selected_topic if isinstance(topic, TopicSelectionResult) else topic
        if selected is None:
            raise ValueError("A selected topic is required")
        return _TopicData(
            title=selected.title,
            summary=selected.summary,
            source_url=selected.url,
            source_name=selected.source,
            category=selected.category,
            analysis=selected.llm_analysis,
        )

    @staticmethod
    def _validate_content_safety(content: str) -> None:
        lowered = content.lower()
        forbidden = ("AIza", "sk-", "gemini_api_key", "linkedin_client_secret", "begin private key")
        if any(marker.lower() in lowered for marker in forbidden):
            raise PostGenerationError("Draft contains a possible secret")
        if any(marker in lowered for marker in ("lorem ipsum", "[your text", "todo:")):
            raise PostGenerationError("Draft contains placeholder text")
        words = re.findall(r"\b\w+\b", lowered)
        if len(words) > 30 and len(set(words)) / len(words) < 0.35:
            raise PostGenerationError("Draft is excessively repetitive")