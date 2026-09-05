"""Deterministic tests for the Phase 8 LangGraph orchestration layer."""

from collections.abc import Sequence

from langgraph.graph import END, START

from config import Settings
from graph.edges import decision_router, route_after_decision
from graph.nodes import DuplicateCheckResult, WorkflowDependencies
from graph.state import WorkflowState
from graph.workflow import build_workflow
from services.fact_checker_service import (
    ClaimCheck,
    ClaimStatus,
    FactCheckResult,
    FactCheckStatus,
)
from services.linkedin_post_service import LinkedInPost
from services.quality_checker_service import QualityCheckResult, QualityStatus
from services.research_service import ResearchArticle
from services.topic_selection_service import TopicCandidate, TopicSelectionResult


def article(title: str = "A new AI model") -> ResearchArticle:
    return ResearchArticle(
        title=title,
        summary="The model combines language and vision capabilities.",
        source="Example Research",
        source_url=f"https://example.com/{title.lower().replace(' ', '-')}",
        category="Multimodal AI",
    )


def topic(title: str = "A new AI model") -> TopicCandidate:
    return TopicCandidate(
        title=title,
        summary="The model combines language and vision capabilities.",
        source="Example Research",
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        category="Multimodal AI",
        importance_score=8,
        freshness_score=9,
        linkedin_score=8,
        technical_relevance=8,
        llm_confidence=0.8,
        audience_value=8,
        novelty_score=10,
        selection_score=8,
    )


def post(title: str = "A new AI model") -> LinkedInPost:
    return LinkedInPost(
        title=title,
        hook="AI progress matters when it changes practical work.",
        body="The model combines language and vision capabilities.",
        key_insights=["The model combines language and vision capabilities."],
        practical_impact="Teams can evaluate this development against practical needs.",
        engagement_question="Which AI workflow would benefit from this development?",
        hashtags=["#AI", "#MultimodalAI", "#ComputerVision"],
        source_url="https://example.com/a-new-ai-model",
        source_name="Example Research",
        topic_category="Multimodal AI",
        content="AI progress matters when it changes practical work.\n\nThe model combines language and vision capabilities.\n\nWhich AI workflow would benefit from this development?\n\n#AI #MultimodalAI #ComputerVision",
    )


def fact_result(status: FactCheckStatus = FactCheckStatus.PASS) -> FactCheckResult:
    claim = ClaimCheck(
        claim="The model combines language and vision capabilities.",
        status=ClaimStatus.SUPPORTED if status == FactCheckStatus.PASS else ClaimStatus.UNCERTAIN,
        evidence="The model combines language and vision capabilities.",
        confidence=0.9 if status == FactCheckStatus.PASS else 0.3,
        reason="Source evidence comparison.",
    )
    return FactCheckResult(
        overall_status=status,
        overall_score=10 if status == FactCheckStatus.PASS else 6,
        claims=[claim],
        supported_claims=1 if status == FactCheckStatus.PASS else 0,
        unsupported_claims=0,
        uncertain_claims=0 if status == FactCheckStatus.PASS else 1,
        hallucination_risk=0 if status == FactCheckStatus.PASS else 0.2,
        explanation="Test evidence.",
        recommendations=[] if status == FactCheckStatus.PASS else ["Review the claim."],
    )


def quality_result(status: QualityStatus = QualityStatus.PASS) -> QualityCheckResult:
    return QualityCheckResult(
        overall_score=10 if status == QualityStatus.PASS else 6,
        readability_score=10,
        professionalism_score=10,
        structure_score=10,
        engagement_score=10,
        hashtag_score=10,
        spam_score=10,
        clickbait_score=10,
        issues=[] if status == QualityStatus.PASS else ["Needs revision."],
        recommendations=[] if status == QualityStatus.PASS else ["Improve the draft."],
        status=status,
    )


class ResearchStub:
    def __init__(self, articles: Sequence[ResearchArticle] | None = None, error: bool = False) -> None:
        self.articles = list(articles or [article()])
        self.error = error

    def fetch_articles(self) -> tuple[list[ResearchArticle], list[str]]:
        if self.error:
            raise RuntimeError("research unavailable")
        return self.articles, []


class SelectionStub:
    def __init__(self, selected: TopicCandidate | None = None) -> None:
        self.selected = selected

    def select_best_topic(self, articles: list[ResearchArticle], session: object = None) -> TopicSelectionResult:
        return TopicSelectionResult(selected_topic=self.selected)


class WriterStub:
    def __init__(self, outputs: Sequence[LinkedInPost] | None = None, error: bool = False) -> None:
        self.outputs = list(outputs or [post()])
        self.calls: list[list[str]] = []
        self.error = error

    def generate_post(self, selected: TopicCandidate, feedback: Sequence[str] = ()) -> LinkedInPost:
        self.calls.append(list(feedback))
        if self.error:
            raise RuntimeError("writer unavailable")
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


class FactStub:
    def __init__(self, results: Sequence[FactCheckResult] | None = None, error: bool = False) -> None:
        self.results = list(results or [fact_result()])
        self.calls = 0
        self.error = error

    def check_post(self, generated: LinkedInPost, source: ResearchArticle, analysis: object = None) -> FactCheckResult:
        self.calls += 1
        if self.error:
            raise RuntimeError("fact checker unavailable")
        return self.results[min(self.calls - 1, len(self.results) - 1)]


class QualityStub:
    def __init__(self, results: Sequence[QualityCheckResult] | None = None, error: bool = False) -> None:
        self.results = list(results or [quality_result()])
        self.calls = 0
        self.error = error

    def check_post(self, generated: LinkedInPost) -> QualityCheckResult:
        self.calls += 1
        if self.error:
            raise RuntimeError("quality checker unavailable")
        return self.results[min(self.calls - 1, len(self.results) - 1)]


def dependencies(
    research: ResearchStub | None = None,
    selection: SelectionStub | None = None,
    writer: WriterStub | None = None,
    fact: FactStub | None = None,
    quality: QualityStub | None = None,
    enable_duplicate_check: bool = False,
) -> WorkflowDependencies:
    return WorkflowDependencies(
        research=research or ResearchStub(),
        topic_selection=selection or SelectionStub(topic()),
        writer=writer or WriterStub(),
        fact_checker=fact or FactStub(),
        quality_checker=quality or QualityStub(),
        enable_duplicate_check=enable_duplicate_check,
    )


def test_graph_builds_and_compiles() -> None:
    workflow = build_workflow(dependencies())
    nodes = workflow.get_graph().nodes

    assert START in nodes
    assert END in nodes
    assert {"research_agent", "topic_selection_agent", "linkedin_writer_agent", "fact_checker_agent", "quality_checker_agent", "duplicate_detection_agent", "decision_router"} <= set(nodes)


def test_happy_path_reaches_approved() -> None:
    workflow = build_workflow(dependencies())

    result = workflow.invoke({"run_id": "run-a", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert result["current_status"] == "APPROVED"
    assert result["selected_topic"].title == "A new AI model"
    assert result["generated_post"].status == "draft"
    assert len(result["agent_logs"]) == 6


def test_all_nodes_preserve_state_and_run_id() -> None:
    workflow = build_workflow(dependencies())

    result = workflow.invoke({"run_id": "stable-run", "max_retries": 2, "retry_count": 0, "warnings": [], "errors": [], "agent_logs": []})

    assert result["run_id"] == "stable-run"
    assert len(result["research_articles"]) == 1
    assert result["fact_check_result"] is not None
    assert result["quality_check_result"] is not None
    assert result["duplicate_check_result"].is_duplicate is False


def test_revision_loop_increments_retry_and_passes_feedback() -> None:
    writer = WriterStub([post("Bad draft"), post("Good draft")])
    fact = FactStub([fact_result(FactCheckStatus.NEEDS_REVIEW), fact_result()])
    workflow = build_workflow(dependencies(writer=writer, fact=fact))

    result = workflow.invoke({"run_id": "revision-run", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert result["current_status"] == "APPROVED"
    assert len(writer.calls) == 2
    assert writer.calls[1]
    assert result["retry_count"] == 1


def test_maximum_retries_terminate_as_rejected() -> None:
    writer = WriterStub([post("Bad draft")])
    fact = FactStub([fact_result(FactCheckStatus.NEEDS_REVIEW)])
    workflow = build_workflow(dependencies(writer=writer, fact=fact))

    result = workflow.invoke({"run_id": "retry-limit", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert result["current_status"] == "REJECTED"
    assert len(writer.calls) == 3
    assert route_after_decision(result) == "reject"


def test_rejected_and_failed_routes_terminate() -> None:
    rejected = {"current_status": "REJECTED", "retry_count": 0, "max_retries": 2}
    failed = {"current_status": "FAILED", "retry_count": 0, "max_retries": 2}

    assert route_after_decision(rejected) == "reject"
    assert route_after_decision(failed) == "failed"


def test_empty_research_and_no_topic_are_safe() -> None:
    empty_workflow = build_workflow(dependencies(research=ResearchStub([]), selection=SelectionStub(None)))
    empty_result = empty_workflow.invoke({"run_id": "empty", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})
    assert empty_result["current_status"] == "FAILED"

    no_topic_workflow = build_workflow(dependencies(selection=SelectionStub(None)))
    no_topic_result = no_topic_workflow.invoke({"run_id": "no-topic", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})
    assert no_topic_result["current_status"] == "FAILED"
    assert "generated_post" not in no_topic_result


def test_service_failures_are_captured() -> None:
    workflow = build_workflow(dependencies(research=ResearchStub(error=True)))

    result = workflow.invoke({"run_id": "failure", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert result["current_status"] == "FAILED"
    assert result["errors"] == ["Research failed"]


def test_writer_failure_is_terminal() -> None:
    workflow = build_workflow(dependencies(writer=WriterStub(error=True)))

    result = workflow.invoke({"run_id": "writer-failure", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert result["current_status"] == "FAILED"
    assert "Writer failed" in result["errors"]


def test_fact_checker_failure_is_terminal() -> None:
    workflow = build_workflow(dependencies(fact=FactStub(error=True)))

    result = workflow.invoke({"run_id": "fact-failure", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert result["current_status"] == "FAILED"
    assert "Fact checker failed" in result["errors"]


def test_quality_checker_failure_is_terminal() -> None:
    workflow = build_workflow(dependencies(quality=QualityStub(error=True)))

    result = workflow.invoke({"run_id": "quality-failure", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert result["current_status"] == "FAILED"
    assert "Quality checker failed" in result["errors"]


def test_duplicate_detection_routes_to_revision() -> None:
    workflow = build_workflow(dependencies(enable_duplicate_check=False))
    state: WorkflowState = {
        "current_status": "APPROVED",
        "fact_check_result": fact_result(),
        "quality_check_result": quality_result(),
        "duplicate_check_result": DuplicateCheckResult(is_duplicate=True, similarity_score=0.95, reason="Similar stored post", status="DUPLICATE"),
        "retry_count": 0,
        "max_retries": 2,
    }

    updated = decision_router(state)

    assert updated["current_status"] == "NEEDS_REVISION"
    assert route_after_decision({**state, **updated}) == "revise"


def test_two_runs_have_isolated_run_ids() -> None:
    workflow = build_workflow(dependencies())
    first = workflow.invoke({"run_id": "run-one", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})
    second = workflow.invoke({"run_id": "run-two", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert first["run_id"] == "run-one"
    assert second["run_id"] == "run-two"
    assert first["run_id"] != second["run_id"]


def test_no_publishing_or_secrets_in_graph_state() -> None:
    workflow = build_workflow(dependencies())
    result = workflow.invoke({"run_id": "security", "max_retries": 2, "retry_count": 0, "errors": [], "warnings": [], "agent_logs": []})

    assert not hasattr(dependencies().writer, "publish_post")
    assert not hasattr(dependencies().writer, "publish")
    assert result["generated_post"].status == "draft"
    assert all("api_key" not in str(value).lower() for value in result.values())