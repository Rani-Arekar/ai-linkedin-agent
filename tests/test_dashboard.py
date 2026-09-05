"""Deterministic tests for the dashboard data/service layer."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.database import Base
from database.models import AgentLog, Post, Topic
from frontend.dashboard_service import create_dashboard_service
from services.dashboard_service import DashboardDataService


def make_factory() -> tuple[sessionmaker[Session], object]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine), engine


def test_dashboard_service_initializes_and_handles_empty_database() -> None:
    factory, engine = make_factory()
    service = DashboardDataService(factory)

    overview = service.get_overview()

    assert overview["research_articles"] == 0
    assert overview["generated_drafts"] == 0
    assert service.get_research_articles() == []
    assert service.get_topics() == []
    assert service.get_drafts() == []
    assert service.get_agent_logs() == []
    engine.dispose()


def test_dashboard_reads_topics_drafts_and_logs() -> None:
    factory, engine = make_factory()
    with factory() as session:
        topic = Topic(title="AI topic", summary="Summary", source="Example", source_url="https://example.com/topic", category="AI Research")
        session.add(topic)
        session.flush()
        session.add(Post(topic_id=topic.id, content="Draft", status="draft"))
        session.add(AgentLog(agent_name="research", action="fetch", status="success"))
        session.commit()
    service = DashboardDataService(factory)

    assert len(service.get_research_articles()) == 1
    assert len(service.get_topics()) == 1
    assert len(service.get_drafts()) == 1
    assert len(service.get_agent_logs()) == 1
    assert service.get_overview()["generated_drafts"] == 1
    engine.dispose()


def test_dashboard_filters_topics() -> None:
    factory, engine = make_factory()
    with factory() as session:
        session.add_all([
            Topic(title="Vision", summary="Summary", source="A", source_url="https://example.com/a", category="Computer Vision"),
            Topic(title="Agents", summary="Summary", source="B", source_url="https://example.com/b", category="AI Agents"),
        ])
        session.commit()
    service = DashboardDataService(factory)

    assert [topic.title for topic in service.get_research_articles(category="AI Agents")] == ["Agents"]
    assert [topic.title for topic in service.get_research_articles(source="A")] == ["Vision"]
    engine.dispose()


def test_manual_run_delegates_to_workflow_service() -> None:
    class WorkflowStub:
        def start_workflow(self) -> dict[str, str]:
            return {"current_status": "APPROVED"}

    factory, engine = make_factory()
    service = DashboardDataService(factory, WorkflowStub())

    assert service.run_workflow_now() == {"current_status": "APPROVED"}
    engine.dispose()


def test_dashboard_has_no_publishing_api() -> None:
    factory, engine = make_factory()
    service = DashboardDataService(factory)

    assert not hasattr(service, "publish_post")
    engine.dispose()