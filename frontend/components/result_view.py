"""Presentation helpers for workflow results."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st
from contentblitz.workflow.routing import AUTHORITATIVE_NODES


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


def _status_label(status: str) -> str:
    normalized = str(status).strip().lower()
    labels = {
        "pending": "pending",
        "running": "running",
        "completed": "completed",
        "skipped": "skipped",
        "degraded": "degraded",
        "failed": "failed",
    }
    return labels.get(normalized, "degraded")


def render_node_execution_statuses(node_statuses: Mapping[str, Any]) -> None:
    st.subheader("Node Execution Status")
    for node_name in AUTHORITATIVE_NODES:
        status = _status_label(str(node_statuses.get(node_name, "pending")))
        st.markdown(f"- `{node_name}`: **{status}**")


def render_progress_events(events: list[Mapping[str, Any]]) -> None:
    if not events:
        return
    with st.expander("Progress Events", expanded=False):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            timestamp = str(event.get("timestamp", "")).strip()
            node_name = str(event.get("node_name", "")).strip()
            status = _status_label(str(event.get("status", "degraded")))
            message = str(event.get("message", "")).strip()
            line = " | ".join([item for item in [timestamp, node_name, status] if item])
            if line and message:
                st.caption(f"{line} | {message}")
            elif line:
                st.caption(line)
            elif message:
                st.caption(message)


def render_status_messages(messages: list[str]) -> None:
    if not messages:
        return
    st.subheader("Workflow Messages")
    for message in messages:
        st.info(str(message).strip())


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


def render_partial_outputs(render_payload: Mapping[str, Any]) -> None:
    partial_outputs = render_payload.get("partial_outputs", {})
    if not isinstance(partial_outputs, Mapping):
        partial_outputs = {}

    blog = str(partial_outputs.get("blog", "")).strip()
    linkedin = str(partial_outputs.get("linkedin", "")).strip()
    research = str(partial_outputs.get("research", "")).strip()

    if blog:
        st.subheader("Partial Blog Draft")
        st.markdown(blog)
    if linkedin:
        st.subheader("Partial LinkedIn Draft")
        st.markdown(linkedin)
    if research:
        st.subheader("Research Summary")
        st.markdown(research)

    image_prompts = render_payload.get("image_prompts", [])
    if isinstance(image_prompts, list) and image_prompts:
        st.subheader("Image Prompts")
        for prompt in image_prompts:
            safe_prompt = str(prompt).strip()
            if safe_prompt:
                st.markdown(f"- {safe_prompt}")

    image_outputs = render_payload.get("image_outputs", [])
    if isinstance(image_outputs, list) and image_outputs:
        st.subheader("Image Outputs")
        for item in image_outputs:
            if not isinstance(item, Mapping):
                continue
            url = item.get("url")
            status = _status_label(str(item.get("status", "completed")))
            provider = str(item.get("provider", "")).strip()
            if isinstance(url, str) and url.strip():
                st.markdown(f"- `{status}` | `{provider or 'unknown'}` | {url.strip()}")
            else:
                identifier = str(item.get("id", "")).strip() or "unavailable"
                st.markdown(f"- `{status}` | `{provider or 'unknown'}` | {identifier}")


def render_degraded_and_error_state(render_payload: Mapping[str, Any]) -> None:
    warnings = render_payload.get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            safe_warning = str(warning).strip()
            if safe_warning:
                st.warning(safe_warning)

    errors = render_payload.get("errors", [])
    if not isinstance(errors, list):
        return
    for item in errors:
        if not isinstance(item, Mapping):
            continue
        message = str(item.get("message", "")).strip()
        recoverable = bool(item.get("recoverable", False))
        if not message:
            continue
        if recoverable:
            st.warning(message)
        else:
            st.error(message)


def render_export_status(render_payload: Mapping[str, Any]) -> None:
    export_status = render_payload.get("export_status", {})
    if not isinstance(export_status, Mapping):
        return
    if not bool(export_status.get("requested", False)):
        return

    st.subheader("Export Status")
    export_paths = export_status.get("paths", {})
    if isinstance(export_paths, Mapping) and export_paths:
        st.success("Export artifacts were generated.")
        st.json(dict(export_paths))

    export_errors = export_status.get("errors", [])
    if isinstance(export_errors, list) and export_errors:
        if bool(export_status.get("non_blocking_failure", False)):
            st.warning("Export completed with non-blocking warnings.")
        else:
            st.error("Export failed.")
        for item in export_errors:
            if not isinstance(item, Mapping):
                continue
            message = str(item.get("message", "")).strip()
            if message:
                st.caption(message)
