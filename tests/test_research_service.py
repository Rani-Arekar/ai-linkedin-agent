"""Offline tests for the Phase 3 RSS research service."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from database.database import Base
from database.models import Source, Topic
from services.research_service import ResearchArticle, ResearchService
from services.research_sources import ResearchSource


FEED = b"""
<rss version="2.0"><channel><title>Example AI Research</title>
<item><title>New Multimodal AI Model Improves Visual Reasoning</title>
<description><![CDATA[<p>A new research model combines language and vision capabilities.</p>]]></description>
<link>https://example.com/article</link><pubDate>Tue, 02 Sep 2026 12:00:00 GMT</pubDate>
</item>
<item><title>Machine Learning Operations Guide</title><link>https://example.com/mlops</link>
<category>MLOps</category></item>
<item><description>No title or URL</description></item>
</channel></rss>
"""

SOURCE = ResearchSource("Example AI Research", "https://example.com/feed.xml", "rss", "AI Research")


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as database_session:
        yield database_session
    engine.dispose()


@pytest.fixture
def service() -> ResearchService:
    return ResearchService(
        settings=Settings(max_articles_per_source=10),
        sources=[SOURCE],
        fetcher=lambda source: FEED,
    )


def test_rss_parsing_and_normalization(service: ResearchService) -> None:
    articles, failures = service.fetch_articles()

    assert failures == []
    assert len(articles) == 2
    assert articles[0].title == "New Multimodal AI Model Improves Visual Reasoning"
    assert articles[0].summary == "A new research model combines language and vision capabilities."
    assert articles[0].source_url == "https://example.com/article"
    assert articles[1].published_at is None


def test_invalid_article_rejected(service: ResearchService) -> None:
    assert service.normalize_article({"title": "", "link": "https://example.com"}, SOURCE) is None
    assert service.normalize_article({"title": "AI", "link": "not-a-url"}, SOURCE) is None


def test_keyword_filtering_and_categories(service: ResearchService) -> None:
    relevant = ResearchArticle(
        title="A transformer research paper",
        summary="A neural network benchmark.",
        source="Example",
        source_url="https://example.com/research",
    )
    irrelevant = ResearchArticle(
        title="Weather forecast",
        summary="Sunny tomorrow.",
        source="Example",
        source_url="https://example.com/weather",
    )

    assert service.validate_article(relevant)
    assert not service.validate_article(irrelevant)
    assert service.classify_category(relevant.title, relevant.summary) == "Deep Learning"
    assert service.classify_category("An unrelated headline", "No special terms") == "General AI"


def test_score_calculation_is_deterministic_and_bounded(service: ResearchService) -> None:
    article = ResearchArticle(
        title="New generative AI release",
        summary="A model update.",
        source="Example",
        source_url="https://example.com/new",
        published_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    scores = service.calculate_scores(article)

    assert set(scores) == {"importance_score", "freshness_score", "linkedin_score"}
    assert all(0 <= score <= 10 for score in scores.values())
    assert scores["freshness_score"] == 10


def test_url_and_title_duplicates_are_skipped(service: ResearchService, session: Session) -> None:
    session.add(
        Topic(
            title="Existing Multimodal AI Model",
            summary="Stored topic",
            source="Example",
            source_url="https://example.com/existing/",
            category="Multimodal AI",
        )
    )
    session.commit()
    articles = [
        ResearchArticle(
            title="Different title",
            summary="AI news",
            source="Example",
            source_url="https://example.com/existing",
        ),
        ResearchArticle(
            title="Existing Multimodal AI Model!",
            summary="AI news",
            source="Example",
            source_url="https://example.com/new-title",
        ),
    ]

    unique, skipped = service.remove_duplicates(session, articles)

    assert unique == []
    assert skipped == 2


def test_database_topic_and_source_storage(service: ResearchService, session: Session) -> None:
    article = ResearchArticle(
        title="AI agent release",
        summary="A practical AI agent tool.",
        source=SOURCE.name,
        source_url="https://example.com/agent",
    )

    assert service.save_topics(session, [article]) == 1
    topic = session.scalar(select(Topic))
    source = session.scalar(select(Source))
    assert topic is not None
    assert topic.category == "AI Agents"
    assert source is not None and source.url == article.source_url


def test_run_counts_duplicates_and_stores_topics(service: ResearchService, session: Session) -> None:
    result = service.run(session)

    assert result.articles_found == 2
    assert result.ai_topics_identified == 2
    assert result.duplicates_skipped == 0
    assert result.topics_stored == 2
    assert session.query(Topic).count() == 2


def test_source_failure_does_not_stop_other_sources() -> None:
    failing = ResearchSource("Unavailable", "https://example.com/fail.xml", "rss", "General AI")
    calls: list[str] = []

    def fetch(source: ResearchSource) -> bytes:
        calls.append(source.name)
        if source.name == "Unavailable":
            raise TimeoutError("test timeout")
        return FEED

    service = ResearchService(
        settings=Settings(), sources=[failing, SOURCE], fetcher=fetch
    )
    articles, failures = service.fetch_articles()

    assert calls == ["Unavailable", "Example AI Research"]
    assert failures == ["Unavailable"]
    assert len(articles) == 2


def test_empty_rss_response_returns_no_articles(service: ResearchService) -> None:
    service.fetcher = lambda source: b"<rss version='2.0'><channel></channel></rss>"

    articles, failures = service.fetch_articles()

    assert articles == []
    assert failures == []


def test_enabled_sources_configuration() -> None:
    second = ResearchSource("Second", "https://example.com/second.xml", "rss", "AI Research")
    service = ResearchService(
        settings=Settings(enabled_research_sources="Second"), sources=[SOURCE, second]
    )

    assert service.fetch_sources() == [second]