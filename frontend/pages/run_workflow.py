"""Page for running ContentBlitz orchestration."""

from __future__ import annotations

import streamlit as st

from contentblitz.ui.progress import build_pending_progress_events
from contentblitz.ui.rendering import build_render_payload
from contentblitz.ui.status import (
    apply_optional_node_skips,
    build_status_messages,
    derive_node_statuses,
    summarize_workflow_status,
    workflow_requires_clarification,
)
from frontend.components.result_view import (
    render_degraded_and_error_state,
    render_execution_indicators,
    render_usage_summary,
    render_final_response,
    render_export_status,
    render_node_execution_statuses,
    render_partial_outputs,
    render_progress_events,
    render_result_header,
    render_status_messages,
    render_sources,
)
from frontend.config import FRONTEND_CONFIG
from frontend.services.orchestrator_client import stream_workflow_progress
from frontend.services.submission_options import (
    WorkflowControls,
    build_requested_outputs,
    sanitize_export_formats,
)
from frontend.session import (
    add_history_entry,
    clear_persistence_messages,
    get_execution_status,
    get_last_error,
    get_last_result,
    get_last_submission,
    get_node_statuses,
    get_persistence_messages,
    get_progress_events,
    get_status_messages,
    set_execution_status,
    set_last_error,
    set_last_result,
    set_last_submission,
    set_node_statuses,
    save_persisted_run,
    set_progress_events,
    set_status_messages,
)

_WIDGET_KEY_EXPORT_ENABLED = "cbx_export_enabled"
_WIDGET_KEY_EXPORT_FORMATS = "cbx_export_formats"


def _execution_status_from_result(
    *,
    result: dict[str, object],
    node_statuses: dict[str, str],
) -> str:
    normalized_statuses = apply_optional_node_skips(
        state=result,
        node_statuses=node_statuses,
    )
    clarification_required = workflow_requires_clarification(
        state=result,
        node_statuses=normalized_statuses,
    )
    return summarize_workflow_status(
        normalized_statuses,
        workflow_status=str(result.get("workflow_status", "")).strip(),
        clarification_required=clarification_required,
    )


def _validate_submission_inputs(
    *,
    safe_query: str,
    requested_outputs: list[str],
    export_requested: bool,
    export_formats: list[str],
) -> str:
    if not safe_query:
        return "A prompt is required before running the workflow."
    if not requested_outputs:
        return "Select at least one workflow output before execution."
    if export_requested and not export_formats:
        return "Select at least one export format when Enable Export is checked."
    return ""


def _build_controls() -> WorkflowControls:
    st.subheader("Workflow Controls")
    col1, col2 = st.columns(2)
    include_blog = col1.checkbox("Blog Output", value=True)
    include_linkedin = col1.checkbox("LinkedIn Output", value=True)
    include_research = col2.checkbox(
        "Research Output",
        value=False,
        help="Requests research output. Orchestrator remains authoritative for final routing.",
    )
    include_image = col2.checkbox("Image Output", value=False)

    export_enabled = st.checkbox(
        "Enable Export",
        value=False,
        key=_WIDGET_KEY_EXPORT_ENABLED,
    )
    if not export_enabled and st.session_state.get(_WIDGET_KEY_EXPORT_FORMATS):
        # Clear stale selections immediately when export is disabled so
        # disabled multiselect chips do not linger in the UI.
        st.session_state[_WIDGET_KEY_EXPORT_FORMATS] = []
    selected_export_formats = st.multiselect(
        "Export Formats",
        options=list(FRONTEND_CONFIG.export_formats),
        default=[],
        key=_WIDGET_KEY_EXPORT_FORMATS,
        disabled=not export_enabled,
        help="Choose one or more formats to apply when export is enabled.",
    )
    if not export_enabled:
        st.caption("Selected formats are ignored until Enable Export is checked.")
    return WorkflowControls(
        include_blog=bool(include_blog),
        include_linkedin=bool(include_linkedin),
        include_research=bool(include_research),
        include_image=bool(include_image),
        export_enabled=bool(export_enabled),
        export_formats=sanitize_export_formats(selected_export_formats),
    )


def render() -> None:
    st.header("Run Workflow")
    st.caption(
        "UI options are submitted as workflow preferences through the orchestration service layer. "
        "The orchestrator owns final routing/classification behavior."
    )

    with st.container(border=True):
        query = st.text_area(
            "Prompt",
            value="",
            placeholder=FRONTEND_CONFIG.default_query_placeholder,
            height=120,
        )
        controls = _build_controls()
        run_clicked = st.button("Run ContentBlitz", type="primary")

    if run_clicked:
        safe_query = str(query).strip()
        requested_outputs = build_requested_outputs(controls)
        export_requested = bool(controls.export_enabled)
        export_formats = controls.export_formats if export_requested else []

        set_last_submission(
            {
                "requested_outputs": requested_outputs,
                "export_requested": export_requested,
                "export_formats": export_formats,
            }
        )

        validation_error = _validate_submission_inputs(
            safe_query=safe_query,
            requested_outputs=requested_outputs,
            export_requested=export_requested,
            export_formats=export_formats,
        )
        if validation_error:
            set_execution_status("idle")
            set_last_error(validation_error)
        else:
            try:
                set_execution_status("running")
                progress_events = [
                    {
                        "node_name": event.node_name,
                        "status": event.status,
                        "message": event.message,
                        "timestamp": event.timestamp,
                        "safe_metadata": dict(event.safe_metadata),
                    }
                    for event in build_pending_progress_events()
                ]
                set_progress_events(progress_events)
                node_statuses = derive_node_statuses(progress_events)
                set_node_statuses(node_statuses)
                set_status_messages(["Workflow started."])
                live_progress_container = st.empty()
                with st.spinner("Executing workflow..."):
                    final_result: dict[str, object] = {}
                    for event in stream_workflow_progress(
                        user_query=safe_query,
                        requested_outputs=requested_outputs,
                        export_requested=export_requested,
                        export_formats=export_formats,
                    ):
                        if event.get("type") == "progress":
                            raw_event = event.get("event")
                            if isinstance(raw_event, dict):
                                progress_events.append(raw_event)
                                set_progress_events(progress_events)
                                node_statuses = derive_node_statuses(progress_events)
                                set_node_statuses(node_statuses)
                                with live_progress_container.container():
                                    render_node_execution_statuses(node_statuses)
                        elif event.get("type") == "final":
                            result_payload = event.get("result")
                            if isinstance(result_payload, dict):
                                final_result = result_payload
                                raw_events = event.get("events")
                                if isinstance(raw_events, list):
                                    progress_events = [
                                        item
                                        for item in raw_events
                                        if isinstance(item, dict)
                                    ]
                                    set_progress_events(progress_events)
                                    node_statuses = derive_node_statuses(
                                        progress_events
                                    )
                                    set_node_statuses(node_statuses)
                normalized_node_statuses = apply_optional_node_skips(
                    state=final_result,
                    node_statuses=node_statuses,
                )
                set_node_statuses(normalized_node_statuses)
                ui_workflow_status = _execution_status_from_result(
                    result=final_result,
                    node_statuses=normalized_node_statuses,
                )
                final_result["ui_workflow_status"] = ui_workflow_status
                final_result["ui_node_statuses"] = normalized_node_statuses
                final_result["ui_selected_options"] = {
                    "requested_outputs": requested_outputs,
                    "export_requested": export_requested,
                    "export_formats": export_formats,
                }
                computed_status_messages = build_status_messages(
                    state=final_result,
                    node_statuses=normalized_node_statuses,
                )
                final_result["status_messages"] = computed_status_messages
                final_result["ui_progress_events"] = progress_events

                set_last_result(final_result)
                set_execution_status(ui_workflow_status)
                set_status_messages(computed_status_messages)
                persisted_run_id = save_persisted_run(
                    result=final_result,
                    last_submission={
                        "requested_outputs": requested_outputs,
                        "export_requested": export_requested,
                        "export_formats": export_formats,
                    },
                    progress_events=progress_events,
                    node_statuses=normalized_node_statuses,
                    status_messages=computed_status_messages,
                )
                if persisted_run_id:
                    final_result["run_id"] = persisted_run_id
                    set_last_result(final_result)
                add_history_entry(
                    user_query=safe_query,
                    requested_outputs=requested_outputs,
                    workflow_status=str(
                        final_result.get("workflow_status", "")
                    ).strip(),
                )
            except Exception:
                set_execution_status("failed")
                set_last_error("Workflow execution failed. Check logs for details.")
                set_status_messages(["Workflow execution failed before completion."])

    last_error = get_last_error()
    if last_error:
        st.error(last_error)
    persistence_messages = get_persistence_messages()
    for message in persistence_messages:
        st.warning(message)
    if persistence_messages:
        clear_persistence_messages()

    result = get_last_result()
    execution_status = get_execution_status()
    result_for_indicators = result or {}
    if isinstance(result, dict):
        normalized_indicator_statuses = apply_optional_node_skips(
            state=result,
            node_statuses=get_node_statuses(),
        )
        clarification_required = workflow_requires_clarification(
            state=result,
            node_statuses=normalized_indicator_statuses,
        )
        indicator_workflow_status = summarize_workflow_status(
            normalized_indicator_statuses,
            workflow_status=str(result.get("workflow_status", "")).strip(),
            clarification_required=clarification_required,
        )
        result_for_indicators = dict(result)
        result_for_indicators["ui_workflow_status"] = indicator_workflow_status
    render_execution_indicators(
        execution_status=execution_status, result=result_for_indicators
    )
    progress_events = get_progress_events()
    node_statuses = get_node_statuses()
    if not node_statuses and progress_events:
        node_statuses = derive_node_statuses(progress_events)
    if isinstance(result, dict):
        node_statuses = apply_optional_node_skips(
            state=result, node_statuses=node_statuses
        )
    render_node_execution_statuses(node_statuses)
    render_progress_events(progress_events)
    render_status_messages(get_status_messages())

    with st.expander("Last Submitted Options", expanded=False):
        st.json(get_last_submission())
        if isinstance(result, dict):
            st.caption("Orchestrator Returned Outputs")
            st.json(result.get("requested_outputs", []))

    result = get_last_result()
    if not result:
        st.info("Run the workflow to view outputs.")
        return

    render_payload = build_render_payload(
        state=result,
        node_statuses=node_statuses,
    )
    render_degraded_and_error_state(render_payload)
    render_usage_summary(render_payload)
    render_partial_outputs(render_payload)
    render_result_header(
        {"ui_workflow_status": render_payload.get("workflow_status", "")}
    )
    render_final_response({"final_response": render_payload.get("final_response", "")})
    render_sources({"sources": render_payload.get("sources", [])})
    render_export_status(render_payload)
