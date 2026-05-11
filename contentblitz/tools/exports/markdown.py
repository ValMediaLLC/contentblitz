"""Markdown export document builder and sanitizer."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

_ENV_NAME_RE = re.compile(
    r"OPENAI_API_KEY|SERP_API_KEY|PERPLEXITY_API_KEY",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\b(?:sk|pplx|serp)_[A-Za-z0-9\-_]{8,}\b|\bsk-[A-Za-z0-9\-_]{8,}\b|\bpplx-[A-Za-z0-9\-_]{8,}\b",
    flags=re.IGNORECASE,
)
_NONE_NULL_RE = re.compile(r"\b(?:none|null)\b", flags=re.IGNORECASE)
_MALFORMED_HEADING_RE = re.compile(r"^(#{1,6})([^ #].*)$")

_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
    "  file \"",
)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _normalize_error_message(error: Any) -> str:
    if isinstance(error, Mapping):
        message = _safe_text(error.get("message"))
        recoverable = bool(error.get("recoverable", False))
    else:
        message = _safe_text(error)
        recoverable = False
    if not message:
        return "A recoverable workflow error occurred." if recoverable else "The workflow ended due to an internal error."
    lowered = message.lower()
    if any(marker in lowered for marker in _STACK_TRACE_MARKERS):
        return "A recoverable workflow error occurred." if recoverable else "The workflow ended due to an internal error."
    message = _ENV_NAME_RE.sub("[REDACTED]", message)
    message = _TOKEN_RE.sub("[REDACTED]", message)
    return message


def _source_key(source: Mapping[str, Any], index: int) -> str:
    url = _safe_text(source.get("url")).lower()
    if url:
        return f"url:{url}"
    title = _safe_text(source.get("title")).lower()
    if title:
        return f"title:{title}"
    return f"idx:{index}"


def _dedupe_sources_for_export(sources: Any) -> List[Dict[str, Any]]:
    best_by_key: Dict[str, Dict[str, Any]] = {}
    score_by_key: Dict[str, float] = {}
    order: List[str] = []
    for index, raw in enumerate(_safe_list(sources)):
        if not isinstance(raw, Mapping):
            continue
        source = dict(raw)
        key = _source_key(source, index)
        score = _as_float(source.get("credibility_score"), default=0.0)
        if key not in best_by_key:
            best_by_key[key] = source
            score_by_key[key] = score
            order.append(key)
            continue
        if score > score_by_key.get(key, 0.0):
            best_by_key[key] = source
            score_by_key[key] = score
    return [best_by_key[key] for key in order]


def _sanitize_line(raw_line: str) -> str:
    line = raw_line.rstrip()
    lowered = line.lower()

    if any(marker in lowered for marker in _STACK_TRACE_MARKERS):
        return ""
    if "data:image/" in lowered or "b64_json" in lowered or "base64" in lowered:
        return ""
    if line.strip().startswith("{") and line.strip().endswith("}") and ":" in line:
        return ""
    if line.strip().startswith("[") and line.strip().endswith("]") and ":" in line:
        return ""

    line = _ENV_NAME_RE.sub("[REDACTED]", line)
    line = _TOKEN_RE.sub("[REDACTED]", line)
    line = _NONE_NULL_RE.sub("", line)

    heading_match = _MALFORMED_HEADING_RE.match(line)
    if heading_match:
        line = f"{heading_match.group(1)} {heading_match.group(2).strip()}"

    return line.strip()


def sanitize_markdown_content(markdown_text: str) -> str:
    """Sanitize markdown text for safe export output."""
    raw = _safe_text(markdown_text).replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    sanitized_lines: List[str] = []
    previous_blank = False
    for raw_line in lines:
        line = _sanitize_line(raw_line)
        if not line:
            if previous_blank:
                continue
            sanitized_lines.append("")
            previous_blank = True
            continue
        sanitized_lines.append(line)
        previous_blank = False

    while sanitized_lines and not sanitized_lines[0]:
        sanitized_lines.pop(0)
    while sanitized_lines and not sanitized_lines[-1]:
        sanitized_lines.pop()

    return "\n".join(sanitized_lines)


def _render_workflow_summary(state: Mapping[str, Any]) -> str:
    workflow_status = _safe_text(state.get("workflow_status")) or "unknown"
    routing_decision = _safe_text(state.get("routing_decision")) or "unknown"
    requested_outputs = [
        _safe_text(item).lower()
        for item in _safe_list(state.get("requested_outputs", []))
        if _safe_text(item)
    ]
    requested_outputs_text = ", ".join(requested_outputs) if requested_outputs else "none"
    lines = [
        "## Workflow Summary",
        f"- Workflow Status: `{workflow_status}`",
        f"- Routing Decision: `{routing_decision}`",
        f"- Requested Outputs: {requested_outputs_text}",
    ]
    return "\n".join(lines)


def _render_warnings_section(state: Mapping[str, Any]) -> str:
    warnings: List[str] = []
    for value in _safe_list(state.get("warnings", [])):
        text = _safe_text(value)
        if text:
            warnings.append(text)

    for value in _safe_list(state.get("status_messages", [])):
        text = _safe_text(value)
        if text and text not in warnings:
            warnings.append(text)

    research_data = _safe_dict(state.get("research_data", {}))
    if bool(research_data.get("degraded", False)):
        warnings.append("Research results are degraded and may require manual verification.")

    image_outputs = _safe_list(state.get("image_outputs", []))
    if any(_safe_text(_safe_dict(item).get("status")).lower() == "failed" for item in image_outputs):
        warnings.append("Image generation encountered a recoverable issue.")

    for error in _safe_list(state.get("errors", [])):
        message = _normalize_error_message(error)
        if message:
            warnings.append(message)

    unique_warnings = list(dict.fromkeys(item for item in warnings if item))
    if not unique_warnings:
        return ""
    lines = ["## Warnings"]
    lines.extend([f"- {item}" for item in unique_warnings])
    return "\n".join(lines)


def _render_text_section(title: str, content: str) -> str:
    body = _safe_text(content)
    if not body:
        return ""
    return f"## {title}\n{body}"


def _render_image_prompts(state: Mapping[str, Any]) -> str:
    prompts = [_safe_text(item) for item in _safe_list(state.get("image_prompts", [])) if _safe_text(item)]
    if not prompts:
        return ""
    lines = ["## Image Prompts"]
    lines.extend([f"- {prompt}" for prompt in prompts])
    return "\n".join(lines)


def _render_image_outputs(state: Mapping[str, Any]) -> str:
    outputs = _safe_list(state.get("image_outputs", []))
    lines: List[str] = []
    for raw in outputs:
        item = _safe_dict(raw)
        status = _safe_text(item.get("status")).lower() or "unknown"
        provider = _safe_text(item.get("provider")) or "unknown"
        url = _safe_text(item.get("url"))
        identifier = _safe_text(item.get("id"))
        if url.lower().startswith("data:image/") or "base64" in url.lower():
            continue

        if status == "failed":
            message = _normalize_error_message(item.get("error", {}))
            if not message:
                message = "Image generation encountered a recoverable issue."
            lines.append(f"- `{status}` | `{provider}` | {message}")
            continue

        if url:
            lines.append(f"- `{status}` | `{provider}` | {url}")
        elif identifier:
            lines.append(f"- `{status}` | `{provider}` | {identifier}")
        else:
            lines.append(f"- `{status}` | `{provider}`")

    if not lines:
        return ""
    return "\n".join(["## Image Outputs", *lines])


def _render_sources(state: Mapping[str, Any]) -> str:
    sources = _dedupe_sources_for_export(_safe_list(state.get("sources", [])))
    if not sources:
        return ""
    lines = ["## Sources"]
    for index, source in enumerate(sources, start=1):
        item = _safe_dict(source)
        title = _safe_text(item.get("title")) or f"Source {index}"
        url = _safe_text(item.get("url"))
        snippet = _safe_text(item.get("snippet"))
        citation_available = bool(item.get("citation_available", False)) and bool(url)
        if citation_available:
            lines.append(f"{index}. [{title}]({url})")
        else:
            lines.append(f"{index}. {title}")
        if snippet:
            lines.append(f"   - {snippet}")
    return "\n".join(lines)


def build_markdown_export_document(state: Mapping[str, Any]) -> str:
    """
    Build a structured markdown export document from workflow state.

    Empty sections are intentionally omitted.
    """
    content_drafts = _safe_dict(state.get("content_drafts", {}))
    blog_body = _safe_text(_safe_dict(content_drafts.get("blog", {})).get("body"))
    linkedin_body = _safe_text(_safe_dict(content_drafts.get("linkedin", {})).get("body"))
    research_body = _safe_text(
        _safe_dict(content_drafts.get("research_report", {})).get("body")
    )
    if not research_body:
        research_data = _safe_dict(state.get("research_data", {}))
        research_body = _safe_text(
            research_data.get("synthesized_summary") or research_data.get("summary")
        )

    sections: List[str] = [
        "# ContentBlitz Export",
        _render_workflow_summary(state),
        _render_warnings_section(state),
        _render_text_section("Blog Draft", blog_body),
        _render_text_section("LinkedIn Draft", linkedin_body),
        _render_text_section("Research Report", research_body),
        _render_image_prompts(state),
        _render_image_outputs(state),
        _render_sources(state),
    ]

    materialized_sections = [section for section in sections if _safe_text(section)]
    raw_markdown = "\n\n".join(materialized_sections).strip()
    return sanitize_markdown_content(raw_markdown)
