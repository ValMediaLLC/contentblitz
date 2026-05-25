"""Page for viewing and restoring persisted workflow runs."""

from __future__ import annotations

import html

import streamlit as st

from contentblitz.ui.rendering import build_render_payload
from frontend.components.result_view import (
    render_collapsible_output_sections,
    render_degraded_and_error_state,
)
from frontend.session import (
    clear_persistence_messages,
    get_persistence_messages,
    get_run_history,
    list_persisted_run_summaries,
    load_persisted_run,
    restore_persisted_run,
)


def _normalize_history_status(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "sucess": "success",
        "success": "success",
        "partial success": "partial_success",
        "partial-success": "partial_success",
        "completed_with_warnings": "partial_success",
    }
    return aliases.get(normalized, normalized)


def _summary_label(summary: dict[str, object]) -> str:
    timestamp = str(summary.get("updated_at", "")).strip() or "unknown-time"
    status = _normalize_history_status(summary.get("workflow_status", "")) or "unknown"
    outputs = summary.get("requested_outputs", [])
    query_preview = str(summary.get("user_query_preview", "")).strip()
    return f"{timestamp} | {status} | outputs={outputs} | {query_preview}"


def _history_status_tone_class(status: str) -> str:
    normalized = _normalize_history_status(status)
    if normalized in {"success", "completed"}:
        return "cbx-status-green"
    if normalized in {"failed", "error"}:
        return "cbx-status-red"
    return "cbx-status-orange"


def _history_status_pill_html(status: str) -> str:
    normalized = _normalize_history_status(status) or "unknown"
    label = normalized.replace("_", " ")
    return (
        f'<span class="cbx-status-pill {_history_status_tone_class(normalized)}">'
        '<span class="cbx-status-pill-dot" aria-hidden="true"></span>'
        f'<span class="cbx-status-pill-text">{html.escape(label)}</span>'
        "</span>"
    )


def render() -> None:
    st.header("Run History")
    st.markdown(
        (
            "<p class='cbx-page-intro'>"
            "Review persisted workflow runs, inspect outcomes, and restore state into "
            "the current browser session."
            "</p>"
        ),
        unsafe_allow_html=True,
    )

    persistence_messages = get_persistence_messages()
    for message in persistence_messages:
        st.warning(message)
    if persistence_messages:
        clear_persistence_messages()

    summaries = list_persisted_run_summaries(limit=200)
    if not summaries:
        st.info("No saved workflow runs were found yet.")
    else:
        preview_cards: list[str] = []
        for summary in summaries[:6]:
            workflow_status = str(summary.get("workflow_status", "unknown"))
            updated_at = (
                str(summary.get("updated_at", "")).strip() or "unknown-time"
            )
            query_preview = (
                str(summary.get("user_query_preview", "")).strip()
                or "No query preview available."
            )
            status_pill = _history_status_pill_html(workflow_status)
            updated_at_html = html.escape(updated_at)
            query_preview_html = html.escape(query_preview)
            preview_cards.append(
                (
                    "<article class='cbx-history-card'>"
                    f"<div>{status_pill}</div>"
                    f"<p class='cbx-history-card-meta'>{updated_at_html}</p>"
                    f"<p class='cbx-history-card-query'>{query_preview_html}</p>"
                    "</article>"
                )
            )
        st.markdown(
            (
                "<section class='cbx-history-grid'>"
                + "".join(preview_cards)
                + "</section>"
            ),
            unsafe_allow_html=True,
        )
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
            render_degraded_and_error_state(render_payload)
            progress_events = (
                list(selected_record.get("ui_progress_events", []))
                if isinstance(selected_record.get("ui_progress_events", []), list)
                else []
            )
            render_collapsible_output_sections(
                render_payload=render_payload,
                status_messages=(
                    selected_record.get("status_messages", [])
                    if isinstance(selected_record.get("status_messages", []), list)
                    else []
                ),
                execution_status=str(
                    render_payload.get("workflow_status", "unknown")
                ).strip()
                or "unknown",
                indicator_result={
                    "ui_workflow_status": render_payload.get("workflow_status", ""),
                    "workflow_status": render_payload.get("workflow_status", ""),
                    "routing_decision": selected_record.get("routing_decision", ""),
                },
                node_statuses=node_statuses,
                progress_events=progress_events,
                raw_state=selected_record,
                raw_submission=selected_record.get("ui_selected_options", {}),
            )

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
        status = _normalize_history_status(item.get("workflow_status", ""))
        timestamp_html = html.escape(timestamp or "unknown-time")
        query_html = html.escape(query or "No query provided.")
        outputs_html = html.escape(str(outputs))
        st.markdown(
            (
                "<article class='cbx-history-timeline-card'>"
                f"<div>{_history_status_pill_html(status)}</div>"
                f"<p class='cbx-history-card-meta'>{timestamp_html}</p>"
                f"<p class='cbx-history-card-query'>{query_html}</p>"
                f"<p class='cbx-history-card-meta'>outputs: {outputs_html}</p>"
                "</article>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"- `{timestamp}` | status: `{status or 'unknown'}` | "
            f"outputs: `{outputs}` | query: {query}"
        )
