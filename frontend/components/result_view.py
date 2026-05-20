"""Presentation helpers for workflow results."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from contentblitz.tools.exports.filenames import resolve_export_dir
from contentblitz.ui.error_display import redact_sensitive_text
from contentblitz.ui.observability import build_observability_diagnostics
from contentblitz.ui.progress import normalize_progress_status, validate_node_name

_DEBUG_TRACEBACK_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_EXPORT_MIME_TYPES = {
    "markdown": "text/markdown",
    "html": "text/html",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_EXPORT_SUFFIXES = {
    "markdown": ".md",
    "html": ".html",
    "pdf": ".pdf",
    "docx": ".docx",
}
_IMAGE_MIME_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_STATUS_GREEN = {"complete", "completed", "enabled", "success", "succeeded"}
_STATUS_ORANGE = {
    "disabled",
    "idle",
    "limited",
    "partial_success",
    "pending",
    "running",
    "skipped",
    "waiting",
}
_STATUS_RED = {
    "blocked",
    "budget_exceeded",
    "canceled",
    "cancelled",
    "degraded",
    "error",
    "errored",
    "failed",
    "failure",
    "negative",
}
_ROUTING_PILL_CLASS = "cbx-status-blue"
_STATUS_LABELS = {
    "budget_exceeded": "budget exceeded",
    "partial_success": "partial success",
    "success": "completed",
    "succeeded": "completed",
}
_BLOG_HEADINGS = {"blog", "blog draft", "blog post", "blog output"}
_LINKEDIN_HEADINGS = {
    "linkedin",
    "linkedin draft",
    "linkedin output",
    "linkedin post",
}
_IMAGE_HEADINGS = {"image assets", "image outputs", "images"}
_RUNNING_EVENT_STATUSES = {"running"}
_VISIBLE_NODE_EVENT_STATUSES = {"running", "completed", "degraded", "failed"}
_SHORT_NODE_SECONDS = 0.8
_MEDIUM_NODE_SECONDS = 1.8
_LONG_NODE_SECONDS = 3.2
_WRITER_IMAGE_NODE_NAMES = {
    "blog_writer_node",
    "linkedin_writer_node",
    "image_agent_node",
}


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


def _normalized_status_key(status: Any) -> str:
    return _safe_text(status).lower().replace(" ", "_").replace("-", "_")


def _status_label(status: str) -> str:
    normalized = _normalized_status_key(status)
    if not normalized:
        return "pending"
    return _STATUS_LABELS.get(normalized, normalized.replace("_", " "))


def _status_tone_class(status: Any) -> str:
    normalized = _normalized_status_key(status)
    if normalized in _STATUS_GREEN:
        return "cbx-status-green"
    if normalized in _STATUS_ORANGE:
        return "cbx-status-orange"
    if normalized in _STATUS_RED:
        return "cbx-status-red"
    return "cbx-status-orange"


def _status_pill_html(status: Any) -> str:
    label = _status_label(_safe_text(status))
    return (
        f'<span class="cbx-status-pill {_status_tone_class(status)}">'
        '<span class="cbx-status-pill-dot" aria-hidden="true"></span>'
        f'<span class="cbx-status-pill-text">{html.escape(label)}</span>'
        "</span>"
    )


def _routing_pill_html(value: Any) -> str:
    label = _safe_text(value) or "n/a"
    return (
        f'<span class="cbx-status-pill {_ROUTING_PILL_CLASS}">'
        '<span class="cbx-status-pill-dot" aria-hidden="true"></span>'
        f'<span class="cbx-status-pill-text">{html.escape(label)}</span>'
        "</span>"
    )


def _summary_dot_class(state: Any, *, blue: bool = False) -> str:
    if blue:
        return "cbx-summary-dot-blue"
    normalized = _normalized_status_key(state)
    if normalized == "running":
        return "cbx-summary-dot-running"
    if normalized in {"completed", "complete", "success", "succeeded"}:
        return "cbx-summary-dot-completed"
    if normalized in {"failed", "error", "degraded"}:
        return "cbx-summary-dot-failed"
    return "cbx-summary-dot-idle"


def _summary_card_html(
    label: str,
    value: Any,
    *,
    state: Any = "",
    blue: bool = False,
    max_value_length: int = 32,
) -> str:
    safe_label = _safe_text(label)
    full_value = _safe_text(value) or "n/a"
    display_value = _truncate_display_value(full_value, max_length=max_value_length)
    dot_class = _summary_dot_class(state or full_value, blue=blue)
    return (
        '<div class="cbx-metric-card" '
        f'title="{html.escape(full_value, quote=True)}">'
        f'<div class="cbx-metric-label">{html.escape(safe_label)}</div>'
        '<div class="cbx-summary-value">'
        f'<span class="cbx-summary-dot {dot_class}" aria-hidden="true"></span>'
        f'<span>{html.escape(display_value)}</span>'
        "</div>"
        "</div>"
    )


def _render_compact_cards(
    items: list[tuple[str, Any]],
    *,
    max_value_length: int = 28,
    status_labels: set[str] | None = None,
    blue_labels: set[str] | None = None,
) -> None:
    cards: list[str] = []
    status_label_set = {item.lower() for item in status_labels or set()}
    blue_label_set = {item.lower() for item in blue_labels or set()}
    for raw_label, raw_value in items:
        safe_label = _safe_text(raw_label)
        if not safe_label:
            continue
        full_value = _safe_text(raw_value) or "n/a"
        display_value = _truncate_display_value(
            full_value,
            max_length=max_value_length,
        )
        is_status_value = safe_label.lower() in status_label_set
        title_value = _status_label(full_value) if is_status_value else full_value
        if is_status_value:
            value_html = _status_pill_html(full_value)
        elif safe_label.lower() in blue_label_set:
            value_html = _routing_pill_html(display_value)
        else:
            value_html = (
                f'<div class="cbx-metric-value">'
                f"{html.escape(display_value)}</div>"
            )
        cards.append(
            (
                '<div class="cbx-metric-card" '
                f'title="{html.escape(title_value, quote=True)}">'
                f'<div class="cbx-metric-label">{html.escape(safe_label)}</div>'
                f"{value_html}"
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


def _contains_stack_trace(text: str) -> bool:
    lowered = _safe_text(text).lower()
    if any(marker in lowered for marker in _DEBUG_TRACEBACK_MARKERS):
        return True
    if '  file "' in lowered and " line " in lowered:
        return True
    return False


def _sanitize_debug_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_debug_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_debug_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_debug_value(item) for item in value]
    if isinstance(value, (str, bytes)):
        text = (
            value.decode("utf-8", errors="replace")
            if isinstance(value, bytes)
            else _safe_text(value)
        )
        safe_text = redact_sensitive_text(text)
        if _contains_stack_trace(safe_text):
            return "Internal details were removed."
        return safe_text
    return value


def _contains_base64_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if not lowered:
        return False
    return lowered.startswith("data:image/") or "base64" in lowered


def _single_line_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _safe_text(value)).strip()


def _normalize_heading_text(raw_heading: str) -> str:
    normalized = _safe_text(raw_heading).strip().strip("#").strip().lower()
    while normalized.endswith(":"):
        normalized = normalized[:-1].rstrip()
    return normalized


def _strip_sources_sections_for_display(markdown_text: str) -> str:
    text = _safe_text(markdown_text)
    if not text:
        return ""
    lines = text.splitlines()
    in_fenced_block = False
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line.startswith("```"):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block:
            continue
        match = _MARKDOWN_HEADING_RE.match(raw_line)
        if not match:
            continue
        heading_text = _normalize_heading_text(match.group(2))
        if heading_text == "sources":
            return "\n".join(lines[:index]).rstrip()
    return text


def _dedupe_leading_repeated_paragraph(markdown_text: str) -> str:
    text = _safe_text(markdown_text)
    if not text:
        return ""
    paragraphs = re.split(r"\n{2,}", text)
    if len(paragraphs) < 2:
        return text
    first = paragraphs[0].strip()
    second = paragraphs[1].strip()
    if not first or first != second:
        return text
    return "\n\n".join([first, *paragraphs[2:]]).strip()


def _render_section(title: str, body: str) -> None:
    safe_title = _safe_text(title)
    safe_body = _strip_sources_sections_for_display(
        _strip_wrapping_markdown_fence(body)
    )
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


def _canonical_heading_key(heading: str) -> str:
    normalized = _normalize_heading_text(heading)
    if normalized in _BLOG_HEADINGS:
        return "blog"
    if normalized in _LINKEDIN_HEADINGS:
        return "linkedin"
    if normalized.startswith("research"):
        return "research"
    if normalized in _IMAGE_HEADINGS:
        return "image_assets"
    if normalized == "sources":
        return "sources"
    if normalized == "quality warnings":
        return "quality_warnings"
    return "other"


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
        match = _MARKDOWN_HEADING_RE.match(raw_line)
        if not match:
            continue
        heading = match.group(2).strip()
        if _canonical_heading_key(heading) != "other":
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
                rendered.linkedin = _dedupe_leading_repeated_paragraph(body)
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
                rendered.linkedin = _dedupe_leading_repeated_paragraph(content)
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
    st.markdown(
        (
            '<div class="cbx-status-line">'
            '<span class="cbx-status-line-label">Status</span>'
            f"{_status_pill_html(workflow_status)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_execution_indicators(
    *,
    execution_status: str,
    result: Mapping[str, Any] | None,
    progress_events: list[Mapping[str, Any]] | None = None,
) -> None:
    workflow_status = ""
    if isinstance(result, Mapping):
        workflow_status = (
            str(result.get("ui_workflow_status", "")).strip()
            or str(result.get("workflow_status", "")).strip()
        )

    rows = _build_node_execution_rows(list(progress_events or []))
    active_node = _active_node_label(rows)
    active_state = _pipeline_summary_state(rows, execution_status)
    cards = [
        _summary_card_html("Execution", execution_status or "idle"),
        _summary_card_html("Workflow Status", workflow_status or "n/a"),
        _summary_card_html(
            "Active Node",
            active_node,
            state=active_state,
            max_value_length=38,
        ),
    ]
    st.markdown(
        '<div class="cbx-metric-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


@dataclass(frozen=True)
class _NodeExecutionRow:
    node_name: str
    status: str
    progress_percent: int
    elapsed_label: str
    duration_seconds: float


def _parse_event_timestamp(value: Any) -> datetime | None:
    raw = _safe_text(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_elapsed_seconds(seconds: float) -> str:
    safe_seconds = max(0.0, float(seconds))
    return f"{safe_seconds:.1f}s"


def _ensure_visible_elapsed(seconds: float) -> float:
    safe_seconds = max(0.0, float(seconds))
    if 0.0 < safe_seconds < 0.1:
        return 0.1
    return safe_seconds


def _node_duration_seconds(node_name: str) -> float:
    if node_name == "research_agent_node":
        return _LONG_NODE_SECONDS
    if node_name in _WRITER_IMAGE_NODE_NAMES:
        return _MEDIUM_NODE_SECONDS
    if node_name in {"output_assembler_node", "export_node"}:
        return _SHORT_NODE_SECONDS
    return 1.2


def _node_status_icon(status: str) -> str:
    normalized = _normalized_status_key(status)
    if normalized == "running":
        return "◎"
    if normalized in {"completed", "degraded"}:
        return "✓"
    if normalized == "failed":
        return "!"
    return "·"


def _node_elapsed_label(
    *,
    elapsed_seconds: float,
) -> str:
    return _format_elapsed_seconds(elapsed_seconds)


def _pipeline_elapsed_label(rows: list[_NodeExecutionRow]) -> str:
    elapsed_values: list[float] = []
    for row in rows:
        label = row.elapsed_label.removesuffix("s")
        try:
            elapsed_values.append(float(label))
        except ValueError:
            continue
    if not elapsed_values:
        return "0.0s"
    return _format_elapsed_seconds(max(elapsed_values))


def _pipeline_elapsed_label_from_events(
    progress_events: list[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> str:
    meaningful_times: list[datetime] = []
    for event in progress_events:
        if not isinstance(event, Mapping):
            continue
        status = _normalized_status_key(
            normalize_progress_status(_safe_text(event.get("status", "pending")))
        )
        if status not in _VISIBLE_NODE_EVENT_STATUSES:
            continue
        timestamp = _parse_event_timestamp(event.get("timestamp", ""))
        if timestamp is not None:
            meaningful_times.append(timestamp)
    if not meaningful_times:
        return "0.0s"
    start_time = min(meaningful_times)
    rows = _build_node_execution_rows(progress_events)
    has_running = any(
        _normalized_status_key(row.status) == "running" for row in rows
    )
    if has_running and now is not None:
        end_time = now
    else:
        end_time = max(meaningful_times)
    return _format_elapsed_seconds((end_time - start_time).total_seconds())


def _active_node_label(rows: list[_NodeExecutionRow]) -> str:
    running_rows = [
        row for row in rows if _normalized_status_key(row.status) == "running"
    ]
    if running_rows:
        return running_rows[-1].node_name
    if rows:
        return rows[-1].node_name
    return "idle"


def _pipeline_summary_state(
    rows: list[_NodeExecutionRow],
    execution_status: Any,
) -> str:
    if any(_normalized_status_key(row.status) == "running" for row in rows):
        return "running"
    normalized_execution = _normalized_status_key(execution_status)
    if normalized_execution in {"completed", "complete", "success", "succeeded"}:
        return "completed"
    if rows and all(
        _normalized_status_key(row.status) in {"completed", "degraded", "failed"}
        for row in rows
    ):
        return "completed"
    return "idle"


def _node_progress_percent(status: str) -> int:
    normalized = _normalized_status_key(status)
    if normalized in {"completed", "failed", "degraded"}:
        return 100
    if normalized in _RUNNING_EVENT_STATUSES:
        return 55
    return 0


def _node_status_class(status: str) -> str:
    normalized = _normalized_status_key(status)
    if normalized == "completed":
        return "cbx-node-status-green"
    if normalized == "degraded":
        return "cbx-node-status-warning"
    if normalized == "failed":
        return "cbx-node-status-red"
    if normalized == "skipped":
        return "cbx-node-status-neutral"
    if normalized in _RUNNING_EVENT_STATUSES:
        return "cbx-node-status-running"
    return "cbx-node-status-neutral"


def _build_node_execution_rows(
    progress_events: list[Mapping[str, Any]],
    *,
    now: datetime | None = None,
) -> list[_NodeExecutionRow]:
    if not progress_events:
        return []

    records: dict[str, dict[str, Any]] = {}

    for event_index, event in enumerate(progress_events):
        if not isinstance(event, Mapping):
            continue
        try:
            node_name = validate_node_name(_safe_text(event.get("node_name", "")))
        except ValueError:
            continue
        status = _normalized_status_key(
            normalize_progress_status(_safe_text(event.get("status", "pending")))
        )
        if not node_name or not status:
            continue

        if status == "skipped":
            records.pop(node_name, None)
            continue
        if status not in _VISIBLE_NODE_EVENT_STATUSES:
            continue

        event_time = _parse_event_timestamp(event.get("timestamp", ""))
        if node_name not in records:
            records[node_name] = {
                "first_seen": event_index,
                "first_seen_at": event_time,
                "last_seen_at": event_time,
                "status": status,
            }
        else:
            records[node_name]["status"] = status
            if event_time is not None:
                records[node_name]["last_seen_at"] = event_time

    if not records:
        return []

    rows = sorted(
        records.items(),
        key=lambda item: (
            item[1]["first_seen"],
        ),
    )
    def _elapsed_seconds_for_row(index: int) -> float:
        node_name, record = rows[index]
        first_seen_at = (
            record.get("first_seen_at")
            if isinstance(record.get("first_seen_at"), datetime)
            else None
        )
        last_seen_at = (
            record.get("last_seen_at")
            if isinstance(record.get("last_seen_at"), datetime)
            else None
        )
        status = _safe_text(record.get("status", "pending")) or "pending"
        status_key = _normalized_status_key(status)
        fallback_seconds = _node_duration_seconds(node_name)
        terminal_statuses = {"completed", "degraded", "failed"}

        if first_seen_at is None:
            if status_key in terminal_statuses:
                return fallback_seconds
            return 0.0

        if status_key == "running":
            if now is None:
                return 0.0
            return _ensure_visible_elapsed((now - first_seen_at).total_seconds())

        if last_seen_at is not None:
            measured = (last_seen_at - first_seen_at).total_seconds()
            if measured > 0:
                return _ensure_visible_elapsed(measured)

        if status_key in terminal_statuses:
            return fallback_seconds
        return 0.0

    return [
        _NodeExecutionRow(
            node_name=node_name,
            status=_safe_text(record.get("status", "pending")) or "pending",
            progress_percent=_node_progress_percent(
                _safe_text(record.get("status", "pending"))
            ),
            elapsed_label=_node_elapsed_label(
                elapsed_seconds=_elapsed_seconds_for_row(index),
            ),
            duration_seconds=_node_duration_seconds(node_name),
        )
        for index, (node_name, record) in enumerate(rows)
    ]


def render_node_execution_statuses(
    progress_events: list[Mapping[str, Any]],
    *,
    live_timers: bool = False,
    empty_message: str = "No node execution events available yet.",
) -> None:
    st.subheader("Node Execution Status")
    now = datetime.now(UTC) if live_timers else None
    rows = _build_node_execution_rows(progress_events, now=now)
    if not rows:
        st.caption(empty_message)
        if live_timers:
            st.markdown(
                (
                    '<div class="cbx-node-status-panel cbx-node-status-empty">'
                    '<div class="cbx-node-timer">Elapsed 0.0s</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        return

    rendered_rows: list[str] = []
    for row in rows:
        status_class = _node_status_class(row.status)
        progress_style = (
            f"--cbx-node-duration:{row.duration_seconds:.1f}s;"
            f"width:{row.progress_percent}%;"
        )
        progress_modifier = " cbx-node-progress-running" if _normalized_status_key(
            row.status
        ) in _RUNNING_EVENT_STATUSES else ""
        row_modifier = " cbx-node-row-running" if progress_modifier else ""
        icon_class = f"cbx-node-icon-{_normalized_status_key(row.status)}"
        elapsed_modifier = " cbx-node-elapsed-running" if progress_modifier else ""
        rendered_rows.append(
            (
                f'<li class="cbx-node-status-row{row_modifier}">'
                f'<div class="cbx-node-status-icon {icon_class}">'
                f"{_node_status_icon(row.status)}"
                "</div>"
                f'<div class="cbx-node-name">{html.escape(row.node_name)}</div>'
                '<div class="cbx-node-progress-track">'
                f'<div class="cbx-node-progress-fill '
                f'{status_class}{progress_modifier}" '
                f'style="{progress_style}"></div>'
                "</div>"
                '<div class="cbx-node-status-badge">'
                f"{_status_pill_html(row.status)}"
                "</div>"
                f'<div class="cbx-node-elapsed{elapsed_modifier}">'
                f"{html.escape(row.elapsed_label)}</div>"
                "</li>"
            )
        )

    st.markdown(
        (
            '<div class="cbx-node-status-panel">'
            '<ul class="cbx-node-status-list">'
            + "".join(rendered_rows)
            + "</ul>"
            '<div class="cbx-node-timer">'
            f"Elapsed {_pipeline_elapsed_label_from_events(progress_events, now=now)}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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


def render_observability_section() -> None:
    """Render observability diagnostics in the Workflow section."""
    try:
        diagnostics = build_observability_diagnostics()
    except Exception:  # pragma: no cover - defensive UI fallback
        diagnostics = {
            "status": "degraded",
            "status_label": "Degraded",
            "tracing_enabled": False,
            "project_name": "ContentBlitz",
            "endpoint_host": "unknown",
            "last_trace_attempt_label": "Unavailable",
            "note": "Observability diagnostics are temporarily unavailable.",
            "dashboard_instruction": (
                "For trace details, review the LangSmith dashboard manually."
            ),
        }

    status = _safe_text(diagnostics.get("status", "")).lower() or "disabled"
    status_label = (
        _safe_text(diagnostics.get("status_label", "")) or _status_label(status)
    )
    tracing_enabled = (
        "true" if bool(diagnostics.get("tracing_enabled", False)) else "false"
    )
    project_name = _safe_text(diagnostics.get("project_name", "")) or "ContentBlitz"
    endpoint_host = _safe_text(diagnostics.get("endpoint_host", "")) or "unknown"
    trace_attempt = (
        _safe_text(diagnostics.get("last_trace_attempt_label", ""))
        or "Not requested"
    )

    st.subheader("Observability")
    _render_compact_cards(
        [
            ("Observability", status_label),
            ("Tracing Enabled", tracing_enabled),
            ("Project", project_name),
            ("Endpoint Host", endpoint_host),
            ("Last Trace Attempt", trace_attempt),
        ],
        max_value_length=22,
        status_labels={"Observability"},
    )

    note = _safe_text(diagnostics.get("note", ""))
    if note:
        if status == "enabled":
            st.caption(note)
        elif status == "degraded":
            st.warning(note)
        else:
            st.info(note)

    dashboard_instruction = _safe_text(
        diagnostics.get("dashboard_instruction", "")
    )
    if dashboard_instruction:
        st.caption(dashboard_instruction)


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


def _render_provider_degradation_status(render_payload: Mapping[str, Any]) -> None:
    degradation = render_payload.get("degradation_metadata", {})
    if not isinstance(degradation, Mapping):
        return
    text_degraded = bool(degradation.get("text_generation_degraded", False))
    image_degraded = bool(degradation.get("image_generation_degraded", False))
    if not text_degraded and not image_degraded:
        return

    st.warning(
        "OpenAI provider unavailable or quota-limited. "
        "ContentBlitz generated limited fallback outputs."
    )
    provider_status = render_payload.get("provider_status", {})
    if not isinstance(provider_status, Mapping):
        provider_status = {}
    _render_compact_cards(
        [
            (
                "Text Generation",
                _safe_text(provider_status.get("text_generation")) or "degraded",
            ),
            (
                "Image Generation",
                _safe_text(provider_status.get("image_generation")) or "completed",
            ),
            ("Search", _safe_text(provider_status.get("search")) or "completed"),
            ("Export", _safe_text(provider_status.get("export")) or "completed"),
        ],
        max_value_length=24,
        status_labels={"Text Generation", "Image Generation", "Search", "Export"},
    )


def _render_fallback_badges() -> None:
    st.markdown(
        (
            '<div class="cbx-status-line">'
            f"{_status_pill_html('degraded')}"
            f"{_status_pill_html('limited')}"
            f"{_status_pill_html('provider_degraded')}"
            "</div>"
        ),
        unsafe_allow_html=True,
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
    _render_sources_list(sources)


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
        _render_image_outputs(image_outputs)


def _render_image_outputs(image_outputs: list[Mapping[str, Any]]) -> None:
    st.markdown("#### Image Outputs")
    for item in image_outputs:
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        local_path = item.get("local_path")
        if _contains_base64_like(url) or _contains_base64_like(local_path):
            st.warning("A non-renderable image payload was hidden for safety.")
            continue
        status = _status_label(str(item.get("status", "completed")))
        if status == "failed":
            error_payload = item.get("error")
            if isinstance(error_payload, Mapping):
                safe_message = _safe_text(error_payload.get("message", ""))
                if safe_message:
                    st.warning(safe_message)
                    continue
            st.warning("Image generation encountered a recoverable issue.")
            continue
        if isinstance(url, str) and url.strip():
            display_url = url.strip()
            st.image(display_url, caption=f"Generated image ({status})")
        elif isinstance(local_path, str) and local_path.strip():
            display_local_path = local_path.strip()
            resolved_path = _resolve_downloadable_image_path(display_local_path)
            if resolved_path is None:
                st.warning("Generated image file is not available for download.")
                continue
            st.image(str(resolved_path), caption=f"Generated image ({status})")
            _render_image_download(resolved_path)
        else:
            identifier = _safe_text(item.get("id", "")) or "unavailable"
            renderable = bool(item.get("renderable", False))
            if renderable:
                st.markdown(f"- `{status}` | {identifier}")
            else:
                st.markdown(
                    f"- `{status}` | {identifier} "
                    "(non-renderable asset reference)"
                )


def _render_sources_list(sources: list[Mapping[str, Any]]) -> None:
    if not sources:
        return
    source_lines: list[str] = []
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            continue
        title = _safe_text(source.get("title", "")) or f"Source {index}"
        url = _safe_text(source.get("url", ""))
        snippet = _single_line_text(source.get("snippet", ""))
        if url:
            heading = f"{index}. [{title}]({url})"
        else:
            heading = f"{index}. {title}"
        if snippet:
            heading = f"{heading} - {snippet}"
        source_lines.append(heading)
    if source_lines:
        st.markdown("\n".join(source_lines))


def _render_debug_panel(
    *,
    progress_events: list[Mapping[str, Any]],
    raw_state: Mapping[str, Any] | None,
    render_payload: Mapping[str, Any],
    raw_submission: Mapping[str, Any] | None,
) -> None:
    with st.expander("Debug / Advanced", expanded=False):
        if progress_events:
            st.markdown("#### Progress Events")
            for event in progress_events:
                if not isinstance(event, Mapping):
                    continue
                timestamp = _safe_text(event.get("timestamp", ""))
                node_name = _safe_text(event.get("node_name", ""))
                status = _status_label(_safe_text(event.get("status", "pending")))
                message = _safe_text(event.get("message", ""))
                details = " | ".join(
                    [
                        item
                        for item in [timestamp, node_name, status, message]
                        if item
                    ]
                )
                if details:
                    st.caption(details)

        if raw_submission is not None:
            st.markdown("#### Last Submitted Options")
            st.json(_sanitize_debug_value(raw_submission), expanded=True)
        if raw_state is not None:
            st.markdown("#### Raw Workflow State")
            st.json(_sanitize_debug_value(raw_state), expanded=False)
        st.markdown("#### Render Payload")
        st.json(_sanitize_debug_value(render_payload), expanded=False)


def _resolve_downloadable_export_path(raw_path: Any, format_name: str) -> Path | None:
    path_text = _safe_text(raw_path)
    fmt = _safe_text(format_name).lower()
    expected_suffix = _EXPORT_SUFFIXES.get(fmt)
    if not path_text or not expected_suffix:
        return None
    try:
        candidate = Path(path_text)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve()
        )
        export_dir = resolve_export_dir().resolve()
    except OSError:
        return None
    if resolved.parent != export_dir:
        return None
    if resolved.suffix.lower() != expected_suffix:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _resolve_downloadable_image_path(raw_path: Any) -> Path | None:
    path_text = _safe_text(raw_path)
    if not path_text:
        return None
    try:
        candidate = Path(path_text)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve()
        )
        image_dir = (resolve_export_dir() / "images").resolve()
    except OSError:
        return None
    if resolved.suffix.lower() not in _IMAGE_MIME_TYPES:
        return None
    try:
        resolved.relative_to(image_dir)
    except ValueError:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _render_export_download(format_name: str, raw_path: Any) -> None:
    safe_fmt = _safe_text(format_name).lower() or "unknown"
    display_fmt = safe_fmt.upper() if safe_fmt in {"pdf", "html"} else safe_fmt.title()
    resolved_path = _resolve_downloadable_export_path(raw_path, safe_fmt)
    if resolved_path is None:
        st.warning(f"{display_fmt} export file is not available for download.")
        return
    try:
        data = resolved_path.read_bytes()
    except OSError:
        st.warning(f"{display_fmt} export file is not available for download.")
        return
    st.download_button(
        label=f"Download {display_fmt}",
        data=data,
        file_name=resolved_path.name,
        mime=_EXPORT_MIME_TYPES.get(safe_fmt, "application/octet-stream"),
    )


def _render_image_download(resolved_path: Path) -> None:
    try:
        data = resolved_path.read_bytes()
    except OSError:
        st.warning("Generated image file is not available for download.")
        return
    st.download_button(
        label="Download Image",
        data=data,
        file_name=resolved_path.name,
        mime=_IMAGE_MIME_TYPES.get(
            resolved_path.suffix.lower(), "application/octet-stream"
        ),
    )


def render_collapsible_output_sections(
    *,
    render_payload: Mapping[str, Any],
    status_messages: list[str],
    execution_status: str,
    indicator_result: Mapping[str, Any] | None,
    node_statuses: Mapping[str, Any],
    progress_events: list[Mapping[str, Any]],
    raw_state: Mapping[str, Any] | None = None,
    raw_submission: Mapping[str, Any] | None = None,
) -> None:
    rendered = _build_rendered_output_sections(render_payload)
    raw_requested_outputs = []
    if isinstance(raw_state, Mapping):
        raw_requested_outputs = raw_state.get("requested_outputs", [])
    requested_outputs = {
        _safe_text(item).lower() for item in raw_requested_outputs if _safe_text(item)
    }
    has_blog = bool(_safe_text(rendered.blog))
    has_linkedin = bool(_safe_text(rendered.linkedin))
    has_research = bool(_safe_text(rendered.research) or rendered.additional)
    image_prompts = render_payload.get("image_prompts", [])
    image_outputs = render_payload.get("image_outputs", [])
    has_images = (
        bool(image_prompts)
        or bool(image_outputs)
        or ("image" in requested_outputs)
    )
    sources = render_payload.get("sources", [])
    export_status = render_payload.get("export_status", {})

    with st.expander("Workflow", expanded=True):
        render_execution_indicators(
            execution_status=execution_status,
            result=indicator_result,
            progress_events=progress_events,
        )
        render_node_execution_statuses(progress_events)
        render_observability_section()
        _render_provider_degradation_status(render_payload)
        render_status_messages(status_messages)
        render_usage_summary(render_payload)
        render_result_header(
            {"ui_workflow_status": render_payload.get("workflow_status", "")}
        )
        if rendered.summary:
            _render_section("Workflow Summary", rendered.summary)

    if has_blog:
        with st.expander("Blog", expanded=True):
            degradation_payload = render_payload.get("degradation_metadata", {})
            if not isinstance(degradation_payload, Mapping):
                degradation_payload = {}
            if bool(
                degradation_payload.get("text_generation_degraded", False)
            ):
                _render_fallback_badges()
            blog_body = _strip_sources_sections_for_display(
                _strip_wrapping_markdown_fence(rendered.blog)
            )
            if blog_body:
                st.markdown(blog_body)

    if has_linkedin:
        with st.expander("LinkedIn", expanded=False):
            degradation_payload = render_payload.get("degradation_metadata", {})
            if not isinstance(degradation_payload, Mapping):
                degradation_payload = {}
            if bool(
                degradation_payload.get("text_generation_degraded", False)
            ):
                _render_fallback_badges()
            linkedin_body = _strip_sources_sections_for_display(
                _strip_wrapping_markdown_fence(rendered.linkedin)
            )
            if linkedin_body:
                st.markdown(linkedin_body)

    if has_images:
        with st.expander("Images", expanded=False):
            if isinstance(image_prompts, list) and image_prompts:
                prompt_lines = [
                    f"- {_safe_text(item)}"
                    for item in image_prompts
                    if _safe_text(item)
                ]
                if prompt_lines:
                    _render_section("Image Prompts", "\n".join(prompt_lines))
            if isinstance(image_outputs, list) and image_outputs:
                _render_image_outputs(image_outputs)
            elif "image" in requested_outputs:
                st.info("No image outputs are currently available.")

    if has_research:
        with st.expander("Research", expanded=False):
            if rendered.research:
                research_body = _strip_sources_sections_for_display(
                    _strip_wrapping_markdown_fence(rendered.research)
                )
                if research_body:
                    st.markdown(research_body)
            for heading, body in rendered.additional:
                _render_section(heading, body)

    if isinstance(sources, list) and sources:
        with st.expander("Sources", expanded=False):
            _render_sources_list(sources)

    if isinstance(export_status, Mapping) and bool(
        export_status.get("requested", False)
    ):
        with st.expander("Exports", expanded=False):
            render_export_status(render_payload)

    _render_debug_panel(
        progress_events=progress_events,
        raw_state=raw_state,
        render_payload=render_payload,
        raw_submission=raw_submission,
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
            _render_export_download(_safe_text(fmt_name), raw_path)

    export_errors = export_status.get("errors", [])
    export_error_count = (
        int(export_status.get("export_error_count", 0))
        if isinstance(export_status.get("export_error_count"), (int, float))
        and not isinstance(export_status.get("export_error_count"), bool)
        else 0
    )
    export_warning_count = (
        int(export_status.get("export_warning_count", 0))
        if isinstance(export_status.get("export_warning_count"), (int, float))
        and not isinstance(export_status.get("export_warning_count"), bool)
        else 0
    )
    if isinstance(export_errors, list) and export_errors:
        if export_error_count > 0 and bool(
            export_status.get("non_blocking_failure", False)
        ):
            st.warning("Export completed with non-blocking warnings.")
        elif export_error_count > 0:
            st.error("Export failed.")
        elif export_warning_count > 0:
            st.info("Export completed with non-blocking warnings.")
        for item in export_errors:
            if not isinstance(item, Mapping):
                continue
            message = str(item.get("message", "")).strip()
            if message and message.lower() not in {"none", "null"}:
                st.caption(message)
