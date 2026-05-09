"""Page for running ContentBlitz orchestration."""

from __future__ import annotations

from typing import List

import streamlit as st

from frontend.components.result_view import (
    render_final_response,
    render_result_header,
    render_sources,
)
from frontend.config import FRONTEND_CONFIG
from frontend.services.orchestrator_client import run_workflow
from frontend.session import (
    add_history_entry,
    get_last_error,
    get_last_result,
    set_last_error,
    set_last_result,
)


def _collect_requested_outputs(defaults: List[str]) -> List[str]:
    selected = st.multiselect(
        "Requested Outputs",
        options=list(FRONTEND_CONFIG.available_outputs),
        default=defaults,
        help="Select one or more outputs. Leave defaults for a standard content run.",
    )
    return [str(item).strip() for item in selected if str(item).strip()]


def render() -> None:
    st.header("Run Workflow")
    query = st.text_area(
        "Prompt",
        value="",
        placeholder=FRONTEND_CONFIG.default_query_placeholder,
        height=120,
    )
    default_outputs = list(FRONTEND_CONFIG.default_outputs)
    requested_outputs = _collect_requested_outputs(default_outputs)

    run_clicked = st.button("Run ContentBlitz", type="primary")
    if run_clicked:
        safe_query = str(query).strip()
        if not safe_query:
            set_last_error("A prompt is required before running the workflow.")
        else:
            try:
                result = run_workflow(
                    user_query=safe_query,
                    requested_outputs=requested_outputs,
                    export_requested=False,
                )
                set_last_result(result)
                add_history_entry(
                    user_query=safe_query,
                    requested_outputs=requested_outputs,
                    workflow_status=str(result.get("workflow_status", "")).strip(),
                )
            except Exception:
                set_last_error("Workflow execution failed. Check logs for details.")

    last_error = get_last_error()
    if last_error:
        st.error(last_error)

    result = get_last_result()
    if not result:
        st.info("Run the workflow to view outputs.")
        return

    render_result_header(result)
    render_final_response(result)
    render_sources(result)

