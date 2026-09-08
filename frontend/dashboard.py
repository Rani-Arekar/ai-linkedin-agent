"""Streamlit dashboard for observing and manually triggering the workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add project root to Python path so imports such as
# `from config import get_settings` work when Streamlit
# launches this file from the frontend directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from database.database import SessionLocal, init_db
from frontend.dashboard_service import create_dashboard_service


def render_dashboard() -> None:
    """Render the dashboard; workflow execution occurs only after a button click."""

    import streamlit as st

    st.set_page_config(page_title="LinkedIn Agent", page_icon="AI", layout="wide")
    st.title("AI LinkedIn Content Agent")
    st.caption(
    "Research, review, and workflow monitoring. "
    "LinkedIn publishing is disabled until human approval."
    )
    dashboard, scheduler = _services()

    if get_settings().scheduler_enabled and not scheduler.get_scheduler_status()["running"]:
        scheduler.start_scheduler()

    section = st.sidebar.radio(
        "Section",
        ["Overview", "Research Articles", "Topic Selection", "LinkedIn Drafts", "Fact Check / Quality", "Workflow Runs", "Agent Logs", "Scheduler", "LinkedIn Connection"],
    )
    if section == "Overview":
        _render_overview(st, dashboard, scheduler)
    elif section == "Research Articles":
        _render_research(st, dashboard)
    elif section == "Topic Selection":
        _render_topics(st, dashboard)
    elif section == "LinkedIn Drafts":
        _render_drafts(st, dashboard)
    elif section == "Fact Check / Quality":
        _render_evaluation(st, dashboard)
    elif section == "Workflow Runs":
        _render_runs(st, dashboard)
    elif section == "Agent Logs":
        _render_logs(st, dashboard)
    elif section == "LinkedIn Connection":
        _render_linkedin(st, dashboard)
    else:
        _render_scheduler(st, scheduler)


def _services() -> tuple[Any, Any]:
    """Create cached application services without executing the workflow."""

    import streamlit as st

    @st.cache_resource
    def build() -> tuple[Any, Any]:
        init_db()
        return create_dashboard_service(SessionLocal)

    return build()


def _render_overview(st: Any, dashboard: Any, scheduler: Any) -> None:
    overview = dashboard.get_overview()
    st.subheader("Overview")
    columns = st.columns(5)
    for column, (label, key) in zip(
        columns,
        [("Research topics", "research_articles"), ("Drafts", "generated_drafts"), ("Approved", "approved"), ("Needs revision", "needs_revision"), ("Rejected", "rejected")],
    ):
        column.metric(label, overview[key])
    last_run = overview["workflow_status"].get("last_run")
    st.write(f"Workflow status: **{last_run.status if last_run else 'No runs yet'}**")
    st.write(f"Scheduler: **{'Running' if scheduler.get_scheduler_status()['running'] else 'Disabled / stopped'}**")
    if not overview["research_articles"] and not overview["generated_drafts"]:
        st.info("No research or drafts exist yet. Use Workflow Runs to start a manual run.")


def _render_research(st: Any, dashboard: Any) -> None:
    st.subheader("Research Articles")
    topics = dashboard.get_research_articles()
    if not topics:
        st.info("No research articles stored.")
        return
    st.dataframe([{"Title": item.title, "Source": item.source, "URL": item.source_url, "Category": item.category, "Importance": item.importance_score, "Freshness": item.freshness_score, "LinkedIn": item.linkedin_score} for item in topics], width="stretch")


def _render_topics(st: Any, dashboard: Any) -> None:
    st.subheader("Topic Selection")
    topics = dashboard.get_topics()
    if not topics:
        st.info("No selected topics stored.")
        return
    for topic in topics:
        with st.expander(topic.title):
            st.write(f"**{topic.category}** | {topic.source}")
            st.write(topic.summary)
            st.write(f"Importance: {topic.importance_score} | Freshness: {topic.freshness_score} | LinkedIn: {topic.linkedin_score}")
            st.link_button("Open source", topic.source_url)


def _render_drafts(st: Any, dashboard: Any) -> None:
    st.subheader("LinkedIn Drafts")

    st.info(
        "LinkedIn publishing requires human approval. "
        "Review each draft before publishing."
    )

    drafts = dashboard.get_drafts()

    if not drafts:
        st.info("No drafts stored.")
        return

    for draft in drafts:
        with st.expander(
            f"Draft #{draft.id} | {draft.status.upper()}"
        ):
            st.text_area(
                "Content",
                draft.content,
                height=220,
                key=f"draft-{draft.id}",
                disabled=True,
            )

            st.write(
                f"Created: {draft.created_at} | "
                f"Quality: {draft.quality_score} | "
                f"Fact check: {draft.fact_check_status}"
            )

            # -----------------------------------------
            # HUMAN APPROVAL
            # -----------------------------------------
            if draft.status == "APPROVED":
                st.warning(
                    "⚠️ This draft is ready for human approval."
                )

                if st.button(
                    "✅ Approve for Publishing",
                    key=f"approve-{draft.id}",
                    type="primary",
                ):
                    try:
                        dashboard.approve_for_publishing(draft.id)
                        st.success(
                            "Draft approved for LinkedIn publishing."
                        )
                        st.rerun()
                    except Exception as error:
                        st.error(
                            f"Approval failed: {type(error).__name__}"
                        )

            # -----------------------------------------
            # READY TO PUBLISH
            # -----------------------------------------
            if draft.status == "APPROVED_FOR_PUBLISH":
                st.success(
                    "✅ Human approval received. "
                    "This draft is ready for publishing."
                )

                connection = dashboard.get_linkedin_connection()
                publishing_enabled = (
                    get_settings().linkedin_publishing_enabled
                )

                if not publishing_enabled:
                    st.warning(
                        "LinkedIn publishing is currently disabled "
                        "in configuration."
                    )

                elif not connection.connected:
                    st.warning(
                        "Connect LinkedIn before publishing."
                    )

                elif st.button(
                    "🚀 Publish to LinkedIn",
                    key=f"publish-{draft.id}",
                    type="primary",
                ):
                    try:
                        published = dashboard.publish_approved_post(
                            draft.id
                        )

                        st.success(
                            "Published successfully: "
                            f"{published.linkedin_post_id}"
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            f"Publishing failed safely: "
                            f"{type(error).__name__}"
                        )

def _render_evaluation(st: Any, dashboard: Any) -> None:
    st.subheader("Fact Check / Quality")
    evaluation = dashboard.get_latest_evaluation()
    if evaluation is None:
        st.info("No workflow evaluation is available in this dashboard session.")
        return
    st.json(_serialize(evaluation))


def _render_runs(st: Any, dashboard: Any) -> None:
    st.subheader("Workflow Runs")
    if st.button("Run Workflow Now", type="primary"):
        with st.spinner("Running LangGraph workflow..."):
            try:
                state = dashboard.run_workflow_now()
                st.session_state["last_workflow_state"] = state
                st.success(f"Workflow completed with status: {state.get('current_status', 'UNKNOWN')}")
                if state.get("errors"):
                    st.warning("; ".join(state["errors"]))
                if state.get("warnings"):
                    st.info("; ".join(state["warnings"]))
            except Exception as error:
                import traceback

                st.error(
                    f"Workflow failed: {type(error).__name__}: {error}"
                )

                st.code(
                    traceback.format_exc(),
                    language="text",
                )
    runs = dashboard.get_run_history()
    if runs:
        st.dataframe([run.model_dump(mode="json") for run in runs], width="stretch")
    else:
        st.info("No workflow runs yet.")


def _render_logs(st: Any, dashboard: Any) -> None:
    st.subheader("Agent Logs")
    logs = dashboard.get_agent_logs()
    if not logs:
        st.info("No persisted agent logs.")
        return
    st.dataframe([{"Timestamp": log.created_at, "Agent": log.agent_name, "Action": log.action, "Status": log.status, "Error": log.error_message} for log in logs], width="stretch")


def _render_scheduler(st: Any, scheduler: Any) -> None:
    st.subheader("Scheduler")
    status = scheduler.get_scheduler_status()
    st.write(f"Enabled: **{status['enabled']}**")
    st.write(f"Schedule: **{status['schedule_time']} ({status['timezone']})**")
    st.write(f"Next run: **{status['next_run'] or 'Not scheduled'}**")
    st.caption("Scheduling is disabled by default and is configured through environment variables.")


def _render_linkedin(st: Any, dashboard: Any) -> None:
    """Render safe LinkedIn connection controls without displaying tokens."""

    st.subheader("LinkedIn Connection")
    status = dashboard.get_linkedin_connection()
    st.write("Connected" if status.connected else "Not connected")
    if status.account_id:
        st.write(f"Account: {status.account_id}")
    if status.expires_at:
        st.write(f"Token expiration: {status.expires_at}")
    st.write(f"Scopes: {', '.join(status.scopes) or 'None'}")
    if not status.connected:
        if st.button("Connect LinkedIn"):
            try:
                st.link_button("Open LinkedIn authorization", dashboard.get_linkedin_authorization_url())
            except Exception as error:
                st.error(f"Connection unavailable: {type(error).__name__}")
    elif st.button("Disconnect LinkedIn"):
        dashboard.disconnect_linkedin()
        st.success("Local LinkedIn connection removed.")


def _serialize(value: Any) -> Any:
    """Convert Pydantic values nested in the dashboard read model to JSON data."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


if __name__ == "__main__":
    render_dashboard()
