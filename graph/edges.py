"""Deterministic conditional routing for the workflow graph."""

from graph.state import WorkflowState


def route_after_research(state: WorkflowState) -> str:
    """Stop safely when research produced no usable articles."""

    return "end" if state.get("current_status") == "FAILED" or not state.get("research_articles") else "continue"


def route_after_selection(state: WorkflowState) -> str:
    """Do not invoke the writer without a selected topic."""

    return "end" if state.get("current_status") == "FAILED" or state.get("selected_topic") is None else "continue"


def route_after_decision(state: WorkflowState) -> str:
    """Route approval, revision, rejection, and infrastructure failure."""

    status = state.get("current_status", "FAILED")
    if status == "NEEDS_REVISION" and state.get("retry_count", 0) <= state.get("max_retries", 2):
        return "revise"
    if status == "NEEDS_REVISION":
        return "reject"
    if status == "APPROVED":
        return "approved"
    if status == "REJECTED":
        return "reject"
    return "failed"


def decision_router(state: WorkflowState) -> WorkflowState:
    """Apply structured evaluation and duplicate results; never call an LLM."""

    fact = state.get("fact_check_result")
    quality = state.get("quality_check_result")
    duplicate = state.get("duplicate_check_result")

    if fact is None or quality is None or duplicate is None:
        return {
            "current_status": "FAILED",
            "errors": [
                *state.get("errors", []),
                "Decision inputs are incomplete",
            ],
        }  # type: ignore[return-value]

    fact_status = getattr(
        getattr(fact, "overall_status", None),
        "value",
        getattr(fact, "overall_status", None),
    )

    quality_status = getattr(
        getattr(quality, "status", None),
        "value",
        getattr(quality, "status", None),
    )

    feedback = [
        *getattr(fact, "recommendations", []),
        *getattr(quality, "recommendations", []),
    ]

    status_value = "APPROVED"

    if fact_status in {"FAIL", "NEEDS_REVIEW"}:
        status_value = (
            "REJECTED"
            if fact_status == "FAIL"
            else "NEEDS_REVISION"
        )

    if quality_status in {"FAIL", "NEEDS_REVIEW"}:
        status_value = (
            "REJECTED"
            if quality_status == "FAIL"
            else "NEEDS_REVISION"
        )

    if duplicate.is_duplicate:
        status_value = "NEEDS_REVISION"
        feedback.append(duplicate.reason)

    hallucination_risk = getattr(fact, "hallucination_risk", 1)

    if status_value == "APPROVED" and (
        hallucination_risk >= 0.7
        or quality_status == "FAIL"
    ):
        status_value = "REJECTED"

    if (
        status_value == "NEEDS_REVISION"
        and state.get("retry_count", 0) >= state.get("max_retries", 2)
    ):
        status_value = "REJECTED"

    updates = {
        "current_status": status_value,
        "revision_feedback": feedback,
    }

    if status_value == "NEEDS_REVISION":
        updates["retry_count"] = state.get("retry_count", 0) + 1

    return updates  # type: ignore[return-value]