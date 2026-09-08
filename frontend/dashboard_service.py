"""Dashboard service composition and read-model helpers."""

from __future__ import annotations

from collections.abc import Callable
from sqlalchemy.orm import Session

from config import get_settings
from graph.nodes import WorkflowDependencies
from scheduler.scheduler import SchedulerService
from services.dashboard_service import DashboardDataService
from services.fact_checker_service import FactCheckerService
from services.linkedin_post_service import LinkedInPostService
from services.linkedin_api_service import LinkedInAPIService
from services.linkedin_oauth_service import LinkedInOAuthService
from services.quality_checker_service import QualityCheckerService
from services.research_service import ResearchService
from services.topic_selection_service import TopicSelectionService
from services.workflow_service import WorkflowService


def create_dashboard_service(session_factory: Callable[[], Session]) -> tuple[DashboardDataService, SchedulerService]:
    """Compose dashboard, scheduler, and existing workflow services."""

    settings = get_settings()
    dependencies = WorkflowDependencies(
        research=ResearchService(settings=settings),
        topic_selection=TopicSelectionService(settings=settings),
        writer=LinkedInPostService(settings=settings),
        fact_checker=FactCheckerService(),
        quality_checker=QualityCheckerService(settings=settings),
        enable_duplicate_check=settings.workflow_enable_duplicate_check,
    )
    workflow = WorkflowService(
    dependencies,
    settings=settings,
    session_factory=session_factory,
)
    scheduler = SchedulerService(workflow.start_workflow, settings=settings)
    oauth = LinkedInOAuthService(settings=settings)
    api = LinkedInAPIService(settings=settings)
    dashboard = DashboardDataService(session_factory, workflow, scheduler, oauth, api)
    return dashboard, scheduler