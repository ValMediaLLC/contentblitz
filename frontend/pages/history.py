"""Page for viewing and restoring persisted workflow runs."""

from __future__ import annotations

import streamlit as st

from contentblitz.ui.rendering import build_render_payload
from frontend.components.result_view import (
    render_degraded_and_error_state,
    render_export_status,
    render_final_response,
    render_partial_outputs,
    render_result_header,
    render_sources,
    render_status_messages,
    render_usage_summary,
)
from frontend.session import (
    clear_persistence_messages,
    get_persistence_messages,
    get_run_history,
    list_persisted_run_summaries,
    load_persisted_run,
    restore_persisted_run,
)


def _summary_label(summary: dict[str, object]) -> str:
    timestamp = str(summary.get("updated_at", "")).strip() or "unknown-time"
    status = str(summary.get("workflow_status", "")).strip() or "unknown"
    outputs = summary.get("requested_outputs", [])
    query_preview = str(summary.get("user_query_preview", "")).strip()
    return f"{timestamp} | {status} | outputs={outputs} | {query_preview}"


def render() -> None:
    st.header("Run History")

    persistence_messages = get_persistence_messages()
    for message in persistence_messages:
        st.warning(message)
    if persistence_messages:
        clear_persistence_messages()

    summaries = list_persisted_run_summaries(limit=200)
    if not summaries:
        st.info("No saved workflow runs were found yet.")
    else:
        run_ids = [
            str(item.get("run_id", "")).strip()
            for item in summaries
            if str(item.get("run_id", "")).strip()
        ]
        selected_run_id = st.selectbox(
            "Saved Runs",
            options=run_ids,
            format_func=lambda rid: _summary_label(
                next(
                    (
                        item
                        for item in summaries
                        if str(item.get("run_id", "")).strip() == rid
                    ),
                    {},
                )
            ),
        )

        col1, col2 = st.columns([1, 2])
        if col1.button(
            "Restore Selected Run", type="primary", disabled=not bool(selected_run_id)
        ):
            restored, message = restore_persisted_run(selected_run_id)
            if restored:
                st.success(message)
            else:
                st.warning(message)

        selected_record = (
            load_persisted_run(selected_run_id) if selected_run_id else None
        )
        if isinstance(selected_record, dict):
            col2.caption(
                "Run ID: "
                f"`{selected_record.get('run_id', '')}`"
                " | Session ID: "
                f"`{selected_record.get('session_id', '')}`"
            )
            node_statuses = (
                dict(selected_record.get("ui_node_statuses", {}))
                if isinstance(selected_record.get("ui_node_statuses", {}), dict)
                else {}
            )
            render_payload = build_render_payload(
                state=selected_record,
                node_statuses=node_statuses,
            )
            render_status_messages(
                selected_record.get("status_messages", [])
                if isinstance(selected_record.get("status_messages", []), list)
                else []
            )
            render_degraded_and_error_state(render_payload)
            render_usage_summary(render_payload)
            render_partial_outputs(render_payload)
            render_result_header(
                {"ui_workflow_status": render_payload.get("workflow_status", "")}
            )
            render_final_response(render_payload)
            render_sources({"sources": render_payload.get("sources", [])})
            render_export_status(render_payload)

    st.divider()
    st.subheader("Current Browser Session Timeline")
    history = get_run_history()
    if not history:
        st.info("No runs yet in this browser session.")
        return

    for item in history:
        timestamp = str(item.get("timestamp_utc", "")).strip()
        query = str(item.get("user_query", "")).strip()
        outputs = item.get("requested_outputs", [])
        status = str(item.get("workflow_status", "")).strip()
        st.markdown(
            f"- `{timestamp}` | status: `{status or 'unknown'}` | "
            f"outputs: `{outputs}` | query: {query}"
        )
