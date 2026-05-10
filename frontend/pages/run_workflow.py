"""Page for running ContentBlitz orchestration."""

from __future__ import annotations

import streamlit as st

from frontend.components.result_view import (
    render_execution_indicators,
    render_final_response,
    render_result_header,
    render_sources,
)
from frontend.config import FRONTEND_CONFIG
from frontend.services.orchestrator_client import run_workflow
from frontend.services.submission_options import (
    WorkflowControls,
    build_requested_outputs,
    sanitize_export_formats,
)
from frontend.session import (
    add_history_entry,
    get_execution_status,
    get_last_error,
    get_last_result,
    get_last_submission,
    set_execution_status,
    set_last_error,
    set_last_result,
    set_last_submission,
)


def _execution_status_from_result(result: dict[str, object]) -> str:
    workflow_status = str(result.get("workflow_status", "")).strip().lower()
    if workflow_status in {"failed", "error", "error_handled"}:
        return "failed"
    if workflow_status in {"partial_success", "completed_with_warnings"}:
        return "partial_success"
    if workflow_status:
        return "success"
    return "completed"


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

    export_enabled = st.checkbox("Enable Export", value=False)
    selected_export_formats = st.multiselect(
        "Export Formats",
        options=list(FRONTEND_CONFIG.export_formats),
        default=list(FRONTEND_CONFIG.default_export_formats) if export_enabled else [],
        disabled=not export_enabled,
    )
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

    with st.form("workflow_submission_form"):
        query = st.text_area(
            "Prompt",
            value="",
            placeholder=FRONTEND_CONFIG.default_query_placeholder,
            height=120,
        )
        controls = _build_controls()
        run_clicked = st.form_submit_button("Run ContentBlitz", type="primary")

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

        if not safe_query:
            set_execution_status("idle")
            set_last_error("A prompt is required before running the workflow.")
        elif not requested_outputs:
            set_execution_status("idle")
            set_last_error("Select at least one workflow output before execution.")
        else:
            try:
                set_execution_status("running")
                with st.spinner("Executing workflow..."):
                    result = run_workflow(
                        user_query=safe_query,
                        requested_outputs=requested_outputs,
                        export_requested=export_requested,
                        export_formats=export_formats,
                    )
                set_last_result(result)
                set_execution_status(_execution_status_from_result(result))
                add_history_entry(
                    user_query=safe_query,
                    requested_outputs=requested_outputs,
                    workflow_status=str(result.get("workflow_status", "")).strip(),
                )
            except Exception:
                set_execution_status("failed")
                set_last_error("Workflow execution failed. Check logs for details.")

    last_error = get_last_error()
    if last_error:
        st.error(last_error)

    result = get_last_result()
    execution_status = get_execution_status()
    render_execution_indicators(execution_status=execution_status, result=result or {})

    with st.expander("Last Submitted Options", expanded=False):
        st.json(get_last_submission())
        if isinstance(result, dict):
            st.caption("Orchestrator Returned Outputs")
            st.json(result.get("requested_outputs", []))

    result = get_last_result()
    if not result:
        st.info("Run the workflow to view outputs.")
        return

    render_result_header(result)
    render_final_response(result)
    render_sources(result)
