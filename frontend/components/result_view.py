"""Presentation helpers for workflow results."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Mapping

import streamlit as st

from contentblitz.workflow.routing import AUTHORITATIVE_NODES


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truncate_display_value(value: Any, *, max_length: int) -> str:
    text = _safe_text(value)
    if not text:
        return "n/a"
    if max_length <= 0 or len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return f"{text[: max_length - 3].rstrip()}..."


def _render_compact_cards(
    items: list[tuple[str, Any]],
    *,
    max_value_length: int = 28,
) -> None:
    cards: list[str] = []
    for raw_label, raw_value in items:
        safe_label = _safe_text(raw_label)
        if not safe_label:
            continue
        full_value = _safe_text(raw_value) or "n/a"
        display_value = _truncate_display_value(
            full_value,
            max_length=max_value_length,
        )
        cards.append(
            (
                '<div class="cbx-metric-card" '
                f'title="{html.escape(full_value, quote=True)}">'
                f'<div class="cbx-metric-label">{html.escape(safe_label)}</div>'
                f'<div class="cbx-metric-value">{html.escape(display_value)}</div>'
                "</div>"
            )
        )
    if not cards:
        return
    st.markdown(
        '<div class="cbx-metric-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _strip_wrapping_markdown_fence(body: str) -> str:
    stripped = _safe_text(body)
    if not stripped:
        return ""
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    opening = lines[0].strip()
    closing = lines[-1].strip()
    if opening.startswith("```") and closing == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _render_section(title: str, body: str) -> None:
    safe_title = _safe_text(title)
    safe_body = _strip_wrapping_markdown_fence(body)
    if not safe_title or not safe_body:
        return
    st.markdown(f"#### {safe_title}")
    st.markdown(safe_body)


@dataclass
class _RenderedOutputSections:
    summary: str = ""
    blog: str = ""
    linkedin: str = ""
    research: str = ""
    image_assets: str = ""
    additional: list[tuple[str, str]] = field(default_factory=list)


def _extract_markdown_sections(markdown_text: str) -> tuple[str, list[tuple[str, str]]]:
    lines = markdown_text.splitlines()
    headings: list[tuple[int, str]] = []
    in_fenced_block = False

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith("```"):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block:
            continue
        heading = ""
        if line.startswith("## "):
            heading = line[3:].strip()
        elif line.startswith("# "):
            heading = line[2:].strip()
        if heading:
            headings.append((index, heading))

    if not headings:
        return _strip_wrapping_markdown_fence(markdown_text), []

    preface = _strip_wrapping_markdown_fence("\n".join(lines[: headings[0][0]]))
    sections: list[tuple[str, str]] = []
    for idx, (start_index, heading) in enumerate(headings):
        end_index = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = _strip_wrapping_markdown_fence(
            "\n".join(lines[start_index + 1 : end_index])
        )
        if body:
            sections.append((heading, body))
    return preface, sections


def _canonical_heading_key(heading: str) -> str:
    normalized = _safe_text(heading).lower()
    if normalized == "blog draft":
        return "blog"
    if normalized == "linkedin draft":
        return "linkedin"
    if normalized.startswith("research"):
        return "research"
    if normalized == "image assets":
        return "image_assets"
    if normalized == "sources":
        return "sources"
    if normalized == "quality warnings":
        return "quality_warnings"
    return "other"


def _section_key_from_partial_label(label: str) -> str:
    normalized = _safe_text(label).lower()
    if normalized.startswith("blog"):
        return "blog"
    if normalized.startswith("linkedin"):
        return "linkedin"
    return "research"


def _build_rendered_output_sections(
    render_payload: Mapping[str, Any],
) -> _RenderedOutputSections:
    rendered = _RenderedOutputSections()
    final_response = _safe_text(render_payload.get("final_response", ""))

    if final_response:
        preface, sections = _extract_markdown_sections(final_response)
        rendered.summary = preface
        for heading, body in sections:
            key = _canonical_heading_key(heading)
            if key == "blog" and not rendered.blog:
                rendered.blog = body
            elif key == "linkedin" and not rendered.linkedin:
                rendered.linkedin = body
            elif key == "research" and not rendered.research:
                rendered.research = body
            elif key == "image_assets" and not rendered.image_assets:
                rendered.image_assets = body
            elif key in {"sources", "quality_warnings"}:
                continue
            else:
                rendered.additional.append((heading, body))

    raw_sections = render_payload.get("partial_output_sections", [])
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, Mapping):
                continue
            label = _safe_text(item.get("label", ""))
            content = _safe_text(item.get("content", ""))
            if not label or not content:
                continue
            key = _section_key_from_partial_label(label)
            if key == "blog" and not rendered.blog:
                rendered.blog = content
            elif key == "linkedin" and not rendered.linkedin:
                rendered.linkedin = content
            elif key == "research" and not rendered.research:
                rendered.research = content

    return rendered


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

    _render_compact_cards(
        [
            ("Execution", execution_status or "idle"),
            ("Workflow Status", workflow_status or "n/a"),
            ("Routing", routing_decision or "n/a"),
        ],
        max_value_length=32,
    )


def _status_label(status: str) -> str:
    normalized = str(status).strip().lower()
    labels = {
        "pending": "pending",
        "running": "running",
        "success": "completed",
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
    _render_compact_cards(
        [
            ("Estimated Tokens", f"~{estimated_tokens:,}"),
            ("Search Queries", search_queries),
            ("Image Requests", image_requests),
            ("Degraded Ops", degraded_ops),
            ("Retries", retries),
            ("Sources Returned", sources_returned),
            ("Image Failures", image_failures),
            ("Exports Generated", export_count),
            ("Budget State", budget_state),
            ("Estimated Cost Level", cost_level),
        ],
        max_value_length=24,
    )


def render_final_response(result: Mapping[str, Any]) -> None:
    final_response = _safe_text(result.get("final_response", ""))
    rendered = _build_rendered_output_sections(result)
    rendered_any = False

    if rendered.summary:
        _render_section("Workflow Summary", rendered.summary)
        rendered_any = True
    if rendered.blog:
        _render_section("Blog", rendered.blog)
        rendered_any = True
    if rendered.linkedin:
        _render_section("LinkedIn", rendered.linkedin)
        rendered_any = True
    if rendered.research:
        _render_section("Research", rendered.research)
        rendered_any = True

    has_renderable_image_outputs = bool(result.get("image_outputs"))
    if rendered.image_assets and not has_renderable_image_outputs:
        _render_section("Image Assets", rendered.image_assets)
        rendered_any = True

    for heading, body in rendered.additional:
        _render_section(heading, body)
        rendered_any = True

    if not rendered_any and final_response:
        _render_section("Workflow Output", final_response)
        rendered_any = True

    if not rendered_any:
        st.info("No final response is currently available.")


def render_sources(result: Mapping[str, Any]) -> None:
    sources = result.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return
    st.markdown("#### Sources")
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            continue
        title = _safe_text(source.get("title", "")) or f"Source {index}"
        url = source.get("url")
        snippet = _safe_text(source.get("snippet", ""))
        citation_available = bool(source.get("citation_available", False))
        if citation_available and isinstance(url, str) and url.strip():
            st.markdown(f"{index}. [{title}]({url.strip()})")
        else:
            st.markdown(f"{index}. {title}")
        if snippet:
            st.markdown(f"{snippet}")


def render_partial_outputs(render_payload: Mapping[str, Any]) -> None:
    raw_sections = render_payload.get("partial_output_sections", [])
    sections: list[tuple[str, str]] = []
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, Mapping):
                continue
            label = _safe_text(item.get("label", ""))
            content = _safe_text(item.get("content", ""))
            if not label or not content:
                continue
            sections.append((label, content))

    final_response = _safe_text(render_payload.get("final_response", ""))
    if not final_response and sections:
        for label, content in sections:
            _render_section(label, content)

    image_prompts = render_payload.get("image_prompts", [])
    if isinstance(image_prompts, list) and image_prompts:
        prompt_lines: list[str] = []
        for prompt in image_prompts:
            safe_prompt = _safe_text(prompt)
            if safe_prompt:
                prompt_lines.append(f"- {safe_prompt}")
        if prompt_lines:
            _render_section("Image Prompts", "\n".join(prompt_lines))

    image_outputs = render_payload.get("image_outputs", [])
    if isinstance(image_outputs, list) and image_outputs:
        st.markdown("#### Image Outputs")
        for item in image_outputs:
            if not isinstance(item, Mapping):
                continue
            url = item.get("url")
            local_path = item.get("local_path")
            status = _status_label(str(item.get("status", "completed")))
            provider = str(item.get("provider", "")).strip()
            if isinstance(url, str) and url.strip():
                display_url = url.strip()
                st.markdown(f"- `{status}` | `{provider or 'unknown'}` | {display_url}")
                st.image(display_url, caption=f"{provider or 'unknown'} ({status})")
            elif isinstance(local_path, str) and local_path.strip():
                display_local_path = local_path.strip()
                st.markdown(
                    f"- `{status}` | `{provider or 'unknown'}` | {display_local_path}"
                )
                st.image(
                    display_local_path,
                    caption=f"{provider or 'unknown'} ({status})",
                )
            else:
                identifier = _safe_text(item.get("id", "")) or "unavailable"
                renderable = bool(item.get("renderable", False))
                if renderable:
                    st.markdown(
                        f"- `{status}` | `{provider or 'unknown'}` | {identifier}"
                    )
                else:
                    st.markdown(
                        f"- `{status}` | `{provider or 'unknown'}` | {identifier} "
                        "(non-renderable asset reference)"
                    )


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

    st.markdown("#### Export Status")
    export_paths = export_status.get("paths", {})
    if isinstance(export_paths, Mapping) and export_paths:
        st.success("Export artifacts were generated.")
        for fmt_name, raw_path in dict(export_paths).items():
            safe_fmt = _safe_text(fmt_name) or "unknown"
            safe_path = _safe_text(raw_path) or "unavailable"
            st.markdown(f"- `{safe_fmt}`: {safe_path}")

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
