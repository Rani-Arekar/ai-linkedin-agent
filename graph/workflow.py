"""Construction and compilation of the LangGraph content workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.edges import (
    decision_router,
    route_after_decision,
    route_after_research,
    route_after_selection,
)
from graph.nodes import (
    WorkflowDependencies,
    duplicate_detection_agent,
    fact_checker_agent,
    linkedin_writer_agent,
    quality_checker_agent,
    research_agent,
    topic_selection_agent,
)
from graph.state import WorkflowState


def build_workflow(dependencies: WorkflowDependencies):
    """Build and compile the workflow without executing it."""

    graph = StateGraph(WorkflowState)
    graph.add_node("research_agent", lambda state: research_agent(state, dependencies))
    graph.add_node("topic_selection_agent", lambda state: topic_selection_agent(state, dependencies))
    graph.add_node("linkedin_writer_agent", lambda state: linkedin_writer_agent(state, dependencies))
    graph.add_node("fact_checker_agent", lambda state: fact_checker_agent(state, dependencies))
    graph.add_node("quality_checker_agent", lambda state: quality_checker_agent(state, dependencies))
    graph.add_node("duplicate_detection_agent", lambda state: duplicate_detection_agent(state, dependencies))
    graph.add_node("decision_router", decision_router)
    graph.add_edge(START, "research_agent")
    graph.add_conditional_edges("research_agent", route_after_research, {"continue": "topic_selection_agent", "end": END})
    graph.add_conditional_edges("topic_selection_agent", route_after_selection, {"continue": "linkedin_writer_agent", "end": END})
    graph.add_edge("linkedin_writer_agent", "fact_checker_agent")
    graph.add_edge("fact_checker_agent", "quality_checker_agent")
    graph.add_edge("quality_checker_agent", "duplicate_detection_agent")
    graph.add_edge("duplicate_detection_agent", "decision_router")
    graph.add_conditional_edges(
        "decision_router",
        route_after_decision,
        {"approved": END, "revise": "linkedin_writer_agent", "reject": END, "failed": END},
    )
    return graph.compile()