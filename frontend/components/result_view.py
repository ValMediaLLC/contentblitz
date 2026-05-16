"""Presentation helpers for workflow results."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st
from contentblitz.workflow.routing import AUTHORITATIVE_NODES


def render_result_header(result: Mapping[str, Any]) -> None:
    workflow_status = (
        str(result.get("ui_workflow_status", "")).strip()
        or str(result.get("workflow_status", "unknown")).strip()
        or "unknown"
    )
    st.subheader("Workflow Result")
    st.write(f"Status: `{workflow_status}`")


def render_execution_indicators(
    *,
    execution_status: str,
    result: Mapping[str, Any] | None,
) -> None:
    workflow_status = ""
    routing_decision = ""
    if isinstance(result, Mapping):
        workflow_status = (
            str(result.get("ui_workflow_status", "")).strip()
            or str(result.get("workflow_status", "")).strip()
        )
        routing_decision = str(result.get("routing_decision", "")).strip()

    col1, col2, col3 = st.columns(3)
    col1.metric("Execution", execution_status or "idle")
    col2.metric("Workflow Status", workflow_status or "n/a")
    col3.metric("Routing", routing_decision or "n/a")


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
        safe_message = str(message).strip()
        if not safe_message or safe_message.lower() in {"none", "null"}:
            continue
        st.info(safe_message)


def render_usage_summary(render_payload: Mapping[str, Any]) -> None:
    usage = render_payload.get("usage_summary", {})
    if not isinstance(usage, Mapping) or not usage:
        return

    def _safe_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
        return 0

    estimated_tokens = _safe_int(usage.get("estimated_tokens_in")) + _safe_int(
        usage.get("estimated_tokens_out")
    )
    search_queries = _safe_int(usage.get("search_queries"))
    image_requests = _safe_int(usage.get("image_generation_requests"))
    degraded_ops = _safe_int(usage.get("degraded_operations"))
    retries = _safe_int(usage.get("retry_attempts"))
    image_failures = _safe_int(usage.get("image_generation_failures"))
    sources_returned = _safe_int(usage.get("sources_returned"))
    export_count = _safe_int(usage.get("export_generation_count"))
    budget_state = str(usage.get("budget_state", "normal")).strip().lower() or "normal"
    cost_level = (
        str(usage.get("estimated_workflow_cost_level", "low")).strip().lower() or "low"
    )

    st.subheader("Workflow Usage")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Estimated Tokens", f"~{estimated_tokens:,}")
    col2.metric("Search Queries", str(search_queries))
    col3.metric("Image Requests", str(image_requests))
    col4.metric("Degraded Ops", str(degraded_ops))
    col5.metric("Retries", str(retries))

    extra_col1, extra_col2, extra_col3 = st.columns(3)
    extra_col1.metric("Sources Returned", str(sources_returned))
    extra_col2.metric("Image Failures", str(image_failures))
    extra_col3.metric("Exports Generated", str(export_count))

    st.caption(f"Budget State: {budget_state} | Estimated Cost Level: {cost_level}")


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
    mode = str(render_payload.get("partial_output_mode", "none")).strip().lower()
    raw_sections = render_payload.get("partial_output_sections", [])
    sections: list[tuple[str, str]] = []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label", "")).strip()
            content = str(item.get("content", "")).strip()
            if not label or not content:
                continue
            sections.append((label, content))

    if mode == "multi_output" and sections:
        st.subheader("Partial Outputs")
        for label, content in sections:
            st.markdown(f"**{label}**")
            st.markdown(content)
    elif mode == "blog_only" and sections:
        st.subheader("Partial Blog Draft")
        st.markdown(sections[0][1])
    elif mode == "linkedin_only" and sections:
        st.subheader("Partial LinkedIn Draft")
        st.markdown(sections[0][1])
    elif mode == "research_only" and sections:
        st.subheader("Research Summary / Research Report")
        st.markdown(sections[0][1])
    elif len(sections) == 1:
        label, content = sections[0]
        if label.lower().startswith("blog"):
            st.subheader("Partial Blog Draft")
        elif label.lower().startswith("linkedin"):
            st.subheader("Partial LinkedIn Draft")
        else:
            st.subheader("Research Summary / Research Report")
        st.markdown(content)

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
            if safe_warning and safe_warning.lower() not in {"none", "null"}:
                st.warning(safe_warning)

    errors = render_payload.get("errors", [])
    if not isinstance(errors, list):
        return
    for item in errors:
        if not isinstance(item, Mapping):
            continue
        message = str(item.get("message", "")).strip()
        recoverable = bool(item.get("recoverable", False))
        if not message or message.lower() in {"none", "null"}:
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
            if message and message.lower() not in {"none", "null"}:
                st.caption(message)
