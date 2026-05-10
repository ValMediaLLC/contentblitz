"""Presentation helpers for workflow results."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st


def render_result_header(result: Mapping[str, Any]) -> None:
    workflow_status = str(result.get("workflow_status", "unknown")).strip() or "unknown"
    st.subheader("Workflow Result")
    st.write(f"Status: `{workflow_status}`")


def render_execution_indicators(
    *,
    execution_status: str,
    result: Mapping[str, Any] | None,
) -> None:
    workflow_status = ""
    routing_decision = ""
    cost_controls: Mapping[str, Any] = {}
    if isinstance(result, Mapping):
        workflow_status = str(result.get("workflow_status", "")).strip()
        routing_decision = str(result.get("routing_decision", "")).strip()
        raw_cost_controls = result.get("cost_controls", {})
        if isinstance(raw_cost_controls, Mapping):
            cost_controls = raw_cost_controls

    col1, col2, col3 = st.columns(3)
    col1.metric("Execution", execution_status or "idle")
    col2.metric("Workflow Status", workflow_status or "n/a")
    col3.metric("Routing", routing_decision or "n/a")

    if cost_controls:
        with st.expander("Cost Controls", expanded=False):
            st.json(dict(cost_controls))


def render_final_response(result: Mapping[str, Any]) -> None:
    final_response = str(result.get("final_response", "")).strip()
    if final_response:
        st.markdown(final_response)
    else:
        st.info("No final response is currently available.")


def render_sources(result: Mapping[str, Any]) -> None:
    sources = result.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return
    st.subheader("Sources")
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            continue
        title = str(source.get("title", "")).strip() or f"Source {index}"
        url = source.get("url")
        snippet = str(source.get("snippet", "")).strip()
        citation_available = bool(source.get("citation_available", False))
        if citation_available and isinstance(url, str) and url.strip():
            st.markdown(f"{index}. [{title}]({url.strip()})")
        else:
            st.markdown(f"{index}. {title}")
        if snippet:
            st.caption(snippet)
