"""Provider-independent, structured LLM services for article analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from config import Settings, get_settings
from services.research_service import ResearchArticle

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LLMServiceError(RuntimeError):
    """Base error for expected LLM service failures."""


class LLMConfigurationError(LLMServiceError):
    """Raised when the selected provider is not configured correctly."""


class LLMProviderError(LLMServiceError):
    """Raised when a provider cannot complete a request or returns bad data."""

class LLMQuotaError(LLMProviderError):
    """Raised when the LLM provider quota has been exhausted."""

class ArticleSummary(BaseModel):
    """Concise structured summary of an article."""

    summary: str = Field(min_length=1, max_length=4_000)


class ArticleAnalysis(BaseModel):
    """Structured analysis used by later topic-selection workflows."""

    summary: str = Field(min_length=1, max_length=4_000)
    key_points: list[str] = Field(min_length=1, max_length=10)
    important_entities: list[str] = Field(default_factory=list, max_length=20)
    technical_concepts: list[str] = Field(default_factory=list, max_length=20)
    why_it_matters: str = Field(min_length=1, max_length=2_000)
    linkedin_relevance: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=1)


class KeyPoints(BaseModel):
    """Structured extracted article points."""

    key_points: list[str] = Field(min_length=1, max_length=10)


class LLMProvider(Protocol):
    """Interface implemented by concrete LLM providers."""

    def generate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        """Generate and validate one structured response."""


class GeminiProvider:
    """Google Gemini provider using the current ``google-genai`` SDK."""

    def __init__(self, settings: Settings) -> None:
        api_key = settings.gemini_api_key or settings.llm_api_key
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is required for the Gemini provider")

        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout * 1000)),
        )
        self._model = settings.gemini_model or settings.llm_model
        self._temperature = settings.llm_temperature

    def generate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        """Generate JSON with Gemini and validate it against the requested model."""

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=response_model,
                ),
            )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, response_model):
                return parsed
            return response_model.model_validate(json.loads(response.text))
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise LLMProviderError("Gemini returned an invalid structured response") from error
        except Exception as error:
            error_text = str(error).lower()

            if (
                "429" in error_text
                or "resource_exhausted" in error_text
                or "quota" in error_text
            ):
                logger.warning(
                    "Gemini quota exhausted: %s",
                    type(error).__name__,
                )
                raise LLMQuotaError(
                    "Gemini quota exhausted. Please wait for the quota to reset "
                    "or check your billing/plan."
                ) from error

            logger.warning(
                "Gemini request failed: %s",
                type(error).__name__,
            )

            raise LLMProviderError(
                f"Gemini request failed: {type(error).__name__}: {error}"
            ) from error


class MockLLMProvider:
    """Deterministic provider used by tests and local development."""

    def generate(self, prompt: str, response_model: type[ResponseModel]) -> ResponseModel:
        """Return predictable valid output without network calls or credentials."""

        if response_model is ArticleSummary:
            return response_model(summary="The article describes an AI development and its technical significance.")
        if response_model is KeyPoints:
            return response_model(key_points=["The article reports an AI development.", "It describes technical implications."])
        if response_model is ArticleAnalysis:
            return response_model(
                summary="The article describes an AI development and its technical significance.",
                key_points=["The article reports an AI development.", "It describes technical implications."],
                important_entities=[],
                technical_concepts=["artificial intelligence"],
                why_it_matters="It may help technical readers understand a current AI development.",
                linkedin_relevance=7.0,
                confidence=0.8,
            )
        if response_model.__name__ == "LinkedInPost":
            return response_model(
                title="Grounding AI Surrogate Models with Experimental Data",
                hook=(
                    "A CFD-trained deep learning surrogate can reproduce numerical "
                    "predictions accurately, but experimental observations can reveal "
                    "systematic discrepancies."
                ),
                body=(
                    "A recent aerospace study presents an experimentally grounded "
                    "correction framework for a CFD-trained deep learning surrogate "
                    "using wind-tunnel pressure-sensitive paint (PSP) measurements.\n\n"
                    "The GeoTransolver surrogate was trained on 2,300 high-fidelity "
                    "CFD simulations of the NASA CRM wing-body configuration. "
                    "The study reports R² > 0.99 for CFD integrated aerodynamic "
                    "forces and pitching moment, while also showing that the "
                    "surrogate did not match the experimental data directly.\n\n"
                    "To address this discrepancy, the researchers trained a "
                    "correction network using spatially registered PSP measurements "
                    "at Mach 0.70 and 0.85. At Mach 0.85, the correction improved "
                    "agreement with PSP measurements, including the wing suction "
                    "peak, shock location, and pressure recovery.\n\n"
                    "On held-out angles of attack, the grounded surrogate agreed "
                    "with measurements to within 2.3–2.7% of the measured Cp range."
                ),
                key_insights=[
                    "The surrogate was trained on 2,300 high-fidelity CFD simulations.",
                    "The correction network learned CFD-to-experiment discrepancies from PSP measurements.",
                    "Measurements were collected at Mach 0.70 and 0.85 across the studied angle-of-attack range.",
                    "The grounded surrogate agreed with held-out measurements to within 2.3–2.7% of the measured Cp range.",
                ],
                practical_impact=(
                    "The study demonstrates that experimental measurements can be "
                    "used to ground a simulation-trained surrogate by learning "
                    "systematic CFD-to-experiment discrepancies without modifying "
                    "the pretrained surrogate parameters."
                ),
                engagement_question=(
                    "How could experimentally grounded correction methods be used "
                    "to improve surrogate-model reliability in other engineering domains?"
                ),
                hashtags=[
                    "#ArtificialIntelligence",
                    "#MachineLearning",
                    "#DeepLearning",
                    "#AerospaceAI",
                    "#ScientificMachineLearning",
                ],
                source_url="https://arxiv.org/abs/2609.04267",
                source_name="arXiv",
                topic_category="AI Research",
            )
        raise LLMProviderError(f"Mock provider does not support {response_model.__name__}")


class LLMService:
    """Generate validated article summaries, analyses, and key points."""

    def __init__(self, provider: LLMProvider | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or self._create_provider(self.settings)

    @staticmethod
    def _create_provider(settings: Settings) -> LLMProvider:
        if settings.llm_provider.lower() == "mock":
            return MockLLMProvider()
        if settings.llm_provider.lower() == "gemini":
            return GeminiProvider(settings)
        raise LLMConfigurationError(f"Unsupported LLM provider: {settings.llm_provider}")

    def analyze_article(self, article: ResearchArticle) -> ArticleAnalysis:
        """Analyze one article using structured output."""

        return self._generate(article, "article_analysis_prompt.txt", ArticleAnalysis)

    def summarize_article(self, article: ResearchArticle) -> ArticleSummary:
        """Summarize one article using structured output."""

        return self._generate(article, "article_summary_prompt.txt", ArticleSummary)

    def extract_key_points(self, article: ResearchArticle) -> KeyPoints:
        """Extract key points from one article using the analysis prompt."""

        return self._generate(article, "article_analysis_prompt.txt", KeyPoints)

    def generate_structured(
        self, prompt: str, response_model: type[ResponseModel]
    ) -> ResponseModel:
        """Generate a validated structured response for a service-specific prompt."""

        try:
            return self.provider.generate(prompt, response_model)
        except LLMServiceError:
            raise
        except Exception as error:
            logger.warning("LLM provider error: %s", type(error).__name__)
            raise LLMProviderError("LLM provider failed") from error

    def _generate(
        self,
        article: ResearchArticle,
        prompt_name: str,
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        if not article.title.strip() and not article.summary.strip():
            raise ValueError("Article must contain a title or summary")
        prompt = load_prompt(prompt_name) + "\n\nARTICLE:\n" + json.dumps(
            article.model_dump(mode="json"), ensure_ascii=True
        )
        try:
            return self.provider.generate(prompt, response_model)
        except LLMServiceError:
            raise
        except Exception as error:
            logger.warning("LLM provider error: %s", type(error).__name__)
            raise LLMProviderError("LLM provider failed") from error


def load_prompt(name: str) -> str:
    """Load a named prompt from the repository prompt directory."""

    path = (PROMPTS_DIR / name).resolve()
    if path.parent != PROMPTS_DIR.resolve() or path.suffix != ".txt":
        raise ValueError("Invalid prompt name")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise LLMConfigurationError(f"Prompt file is unavailable: {name}") from error