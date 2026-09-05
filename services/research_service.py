"""RSS-based AI research collection and persistence service."""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database import crud
from database.models import Source, Topic
from services.research_sources import DEFAULT_RESEARCH_SOURCES, ResearchSource
from services.retry_service import retry_call

logger = logging.getLogger(__name__)


class ResearchArticle(BaseModel):
    """Validated, normalized article data produced by an RSS feed."""

    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(default="", max_length=10_000)
    source: str = Field(min_length=1, max_length=255)
    source_url: str = Field(min_length=1, max_length=2048)
    published_at: datetime | None = None
    category: str = Field(default="General AI", min_length=1, max_length=100)

    @field_validator("title", "summary", "source", "category")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        normalized = normalize_url(value)
        if not is_valid_url(normalized):
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        return normalized


class ResearchRunResult(BaseModel):
    """Counters and saved topics returned by one research run."""

    articles_found: int = 0
    ai_topics_identified: int = 0
    duplicates_skipped: int = 0
    topics_stored: int = 0
    failed_sources: list[str] = Field(default_factory=list)


def normalize_url(value: str) -> str:
    """Normalize URL casing, fragments, and a trailing slash for comparison."""

    parts = urlsplit(value.strip())
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def is_valid_url(value: str) -> bool:
    """Return whether a URL uses HTTP(S) and has a hostname."""

    parts = urlsplit(value)
    return parts.scheme in {"http", "https"} and bool(parts.netloc)


class ResearchService:
    """Collect, validate, classify, score, deduplicate, and save RSS articles."""

    def __init__(
        self,
        settings: Settings | None = None,
        sources: Sequence[ResearchSource] | None = None,
        fetcher: Callable[[ResearchSource], bytes] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.sources = tuple(sources or DEFAULT_RESEARCH_SOURCES)
        self.fetcher = fetcher or self._fetch_feed
        self.keywords = tuple(
            keyword.strip().lower()
            for keyword in self.settings.research_keywords.split(",")
            if keyword.strip()
        )

    def fetch_sources(self) -> list[ResearchSource]:
        """Return enabled RSS sources, optionally filtered by environment configuration."""

        enabled = self.settings.enabled_research_sources
        if not enabled:
            return list(self.sources)
        names = {item.strip().lower() for item in enabled.split(",") if item.strip()}
        return [source for source in self.sources if source.name.lower() in names]

    def fetch_articles(self) -> tuple[list[ResearchArticle], list[str]]:
        """Fetch and normalize articles while continuing after source failures."""

        articles: list[ResearchArticle] = []
        failed_sources: list[str] = []
        for source in self.fetch_sources():
            try:
                parsed = feedparser.parse(self.fetcher(source))
                entries = parsed.entries[: self.settings.max_articles_per_source]
                for entry in entries:
                    article = self.normalize_article(entry, source)
                    if article is not None:
                        articles.append(article)
                logger.info("Source fetched: %s (%d articles)", source.name, len(entries))
            except (httpx.HTTPError, TimeoutError, ValueError, OSError) as error:
                logger.warning("Source unavailable: %s (%s)", source.name, error)
                failed_sources.append(source.name)
            except Exception:
                logger.exception("Unexpected RSS parsing failure for %s", source.name)
                failed_sources.append(source.name)
        return articles, failed_sources

    def normalize_article(self, entry: Any, source: ResearchSource) -> ResearchArticle | None:
        """Convert a feed entry into a validated article, or reject it with a warning."""

        title = clean_text(entry.get("title", ""))
        url = normalize_url(str(entry.get("link", "")))
        if not title or not is_valid_url(url):
            logger.warning("Invalid article from %s", source.name)
            return None
        summary = clean_text(entry.get("summary", entry.get("description", "")))[:10_000]
        published_at = parse_published_at(entry)
        category = clean_text(entry.get("category", "")) or source.category
        try:
            return ResearchArticle(
                title=title,
                summary=summary,
                source=source.name,
                source_url=url,
                published_at=published_at,
                category=self.classify_category(title, summary, category),
            )
        except ValueError:
            logger.warning("Invalid article from %s", source.name)
            return None

    def validate_article(self, article: ResearchArticle) -> bool:
        """Return whether an article contains meaningful AI-relevant content."""

        text = f"{article.title} {article.summary}".lower()
        return bool(article.title and article.source_url and any(keyword in text for keyword in self.keywords))

    def classify_category(self, title: str, summary: str, fallback: str = "General AI") -> str:
        """Assign the first matching deterministic category from specific to broad rules."""

        text = f"{title} {summary}".lower()
        rules = (
            ("Generative AI", ("generative ai", "text generation", "image generation")),
            ("Large Language Models", ("large language model", "llm", "language model")),
            ("Computer Vision", ("computer vision", "visual reasoning", "image recognition")),
            ("Multimodal AI", ("multimodal", "vision-language")),
            ("AI Agents", ("ai agent", "agentic", "autonomous agent")),
            ("Robotics", ("robotics", "humanoid robot")),
            ("MLOps", ("mlops", "model deployment", "model monitoring")),
            ("Responsible AI", ("responsible ai", "ai safety", "ai governance")),
            ("Deep Learning", ("deep learning", "neural network")),
            ("Machine Learning", ("machine learning", "ml model")),
            ("NLP", ("natural language processing", "nlp")),
            ("AI Research", ("research paper", "arxiv", "benchmark")),
            ("AI Tools", ("ai tool", "developer tool", "open source")),
        )
        for category, terms in rules:
            if any(term in text for term in terms):
                return category
        return fallback if fallback in {rule[0] for rule in rules} else "General AI"

    def calculate_scores(self, article: ResearchArticle) -> dict[str, float]:
        """Calculate transparent preliminary 0-10 scores; these are not LLM judgments."""

        age_days = (
            max((datetime.now(timezone.utc) - article.published_at).total_seconds() / 86400, 0)
            if article.published_at
            else 31
        )
        freshness = 10.0 if age_days <= 1 else 8.0 if age_days <= 3 else 6.0 if age_days <= 7 else 3.0 if age_days <= 30 else 1.0
        importance = 7.0 if article.category != "General AI" else 5.0
        if any(term in article.title.lower() for term in ("launch", "release", "breakthrough", "new")):
            importance = min(10.0, importance + 1.0)
        linkedin = min(10.0, round((importance + freshness) / 2 + (1 if article.summary else 0), 1))
        return {"importance_score": importance, "freshness_score": freshness, "linkedin_score": linkedin}

    def remove_duplicates(
        self, session: Session, articles: Iterable[ResearchArticle]
    ) -> tuple[list[ResearchArticle], int]:
        """Remove duplicate URLs and highly similar normalized titles against DB and batch data."""

        existing = list(session.execute(select(Topic.source_url, Topic.title)))
        existing_urls = {normalize_url(row[0]) for row in existing}
        existing_titles = [normalize_title(row[1]) for row in existing]
        unique: list[ResearchArticle] = []
        skipped = 0
        for article in articles:
            title = normalize_title(article.title)
            if normalize_url(article.source_url) in existing_urls or any(
                SequenceMatcher(None, title, other).ratio() >= 0.92
                for other in existing_titles
            ) or any(
                SequenceMatcher(None, title, normalize_title(item.title)).ratio() >= 0.92
                for item in unique
            ):
                logger.warning("Duplicate article skipped: %s", article.title)
                skipped += 1
                continue
            unique.append(article)
            existing_urls.add(normalize_url(article.source_url))
            existing_titles.append(title)
        return unique, skipped

    def save_topics(self, session: Session, articles: Iterable[ResearchArticle]) -> int:
        """Persist topics and upsert their source records using the existing database."""

        stored = 0
        try:
            for article in articles:
                classified = article.model_copy(
                    update={
                        "category": self.classify_category(
                            article.title, article.summary, article.category
                        )
                    }
                )
                crud.create_topic(
                    session,
                    **classified.model_dump(exclude={"published_at"}),
                    **self.calculate_scores(classified),
                )
                source = session.scalar(select(Source).where(Source.url == classified.source_url))
                if source is None:
                    crud.create_source(
                        session,
                        name=classified.source,
                        url=classified.source_url,
                        source_type="rss",
                        last_checked=datetime.now(timezone.utc),
                    )
                else:
                    source.last_checked = datetime.now(timezone.utc)
                    session.commit()
                stored += 1
            logger.info("Topics stored: %d", stored)
            return stored
        except SQLAlchemyError:
            session.rollback()
            logger.exception("Database failure while storing research topics")
            raise

    def run(self, session: Session) -> ResearchRunResult:
        """Execute one complete research pass and persist valid, unique AI topics."""

        logger.info("Research started")
        articles, failed_sources = self.fetch_articles()
        relevant = [article for article in articles if self.validate_article(article)]
        unique, duplicates = self.remove_duplicates(session, relevant)
        stored = self.save_topics(session, unique)
        return ResearchRunResult(
            articles_found=len(articles),
            ai_topics_identified=len(relevant),
            duplicates_skipped=duplicates,
            topics_stored=stored,
            failed_sources=failed_sources,
        )

    def _fetch_feed(self, source: ResearchSource) -> bytes:
        response = retry_call(
            lambda: httpx.get(
                source.url,
                timeout=self.settings.research_timeout_seconds,
                follow_redirects=True,
            ),
            attempts=3,
            initial_delay=0.1,
            retry_exceptions=(httpx.TimeoutException, httpx.ConnectError),
        )
        response.raise_for_status()
        return response.content


def clean_text(value: str) -> str:
    """Remove markup and normalize whitespace from feed text."""

    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(str(value)))
    return re.sub(r"\s+", " ", without_tags).strip()


def normalize_title(value: str) -> str:
    """Normalize a title for lightweight duplicate comparison."""

    return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()


def parse_published_at(entry: Any) -> datetime | None:
    """Parse common RSS/Atom date representations without requiring publication dates."""

    for key in ("published", "updated", "created"):
        value = entry.get(key)
        if not value:
            continue
        try:
            parsed = parsedate_to_datetime(str(value))
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError, OverflowError):
            pass
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pass
    return None


if __name__ == "__main__":
    logging.basicConfig(level=get_settings().log_level)
    from database.database import SessionLocal, init_db

    init_db()
    with SessionLocal() as db_session:
        result = ResearchService().run(db_session)
    print(f"Articles found: {result.articles_found}")
    print(f"AI topics identified: {result.ai_topics_identified}")
    print(f"Duplicates skipped: {result.duplicates_skipped}")
    print(f"Topics stored: {result.topics_stored}")