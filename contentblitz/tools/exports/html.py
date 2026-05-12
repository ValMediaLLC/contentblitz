"""HTML export renderer and sanitizer for ContentBlitz."""

from __future__ import annotations

from html import escape
import re
from typing import Any, Dict, List, Mapping

from contentblitz.quality.citations import validate_citation_sources
from contentblitz.safety.output_sanitizer import (
    sanitize_html_output,
    sanitize_plain_output,
)
from contentblitz.tools.exports.markdown import (
    collect_export_warnings,
    derive_export_workflow_status,
)

_ENV_NAME_RE = re.compile(
    r"OPENAI_API_KEY|SERP_API_KEY|PERPLEXITY_API_KEY",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\b(?:sk|pplx|serp)_[A-Za-z0-9\-_]{8,}\b|\bsk-[A-Za-z0-9\-_]{8,}\b|\bpplx-[A-Za-z0-9\-_]{8,}\b",
    flags=re.IGNORECASE,
)
_NONE_NULL_RE = re.compile(r"\b(?:none|null)\b", flags=re.IGNORECASE)
_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
    "  file \"",
)
_SCRIPT_TAG_RE = re.compile(r"(?is)<script\b[^>]*>.*?</script>")
_UNSAFE_TAG_RE = re.compile(r"(?is)</?(?:iframe|object|embed)\b[^>]*>")
_EVENT_HANDLER_ATTR_RE = re.compile(
    r"""(?is)\s+on[a-z0-9_-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)"""
)
_JAVASCRIPT_URL_RE = re.compile(r"(?i)javascript:")
_GENERIC_IMAGE_ERROR_RECOVERABLE = "Image generation encountered a recoverable issue."
_GENERIC_IMAGE_ERROR_FATAL = "Image generation failed safely."


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _sanitize_plain_text(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        return ""
    lowered = raw.lower()
    if any(marker in lowered for marker in _STACK_TRACE_MARKERS):
        return ""
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        return ""
    redacted = _ENV_NAME_RE.sub("[REDACTED]", raw)
    redacted = _TOKEN_RE.sub("[REDACTED]", redacted)
    redacted = _NONE_NULL_RE.sub("", redacted)
    sanitized, _ = sanitize_plain_output(redacted)
    return sanitized.strip()


def _safe_href(value: Any) -> str:
    url = _safe_text(value)
    if not url:
        return ""
    lowered = url.lower()
    if lowered.startswith(("http://", "https://")):
        return escape(url, quote=True)
    return ""


def _normalize_image_error_message(error: Any) -> str:
    recoverable = True
    message = ""
    code = ""
    provider = ""
    if isinstance(error, Mapping):
        recoverable = bool(error.get("recoverable", True))
        message = _safe_text(error.get("message"))
        code = _safe_text(error.get("code")).lower()
        provider = _safe_text(error.get("provider")).lower()
    else:
        message = _safe_text(error)

    lowered = message.lower()
    suspicious = (
        not message
        or "{'code':" in lowered
        or '"code":' in lowered
        or "configuration_error" in code
        or "configuration_error" in lowered
        or "provider':" in lowered
        or '"provider":' in lowered
        or provider == "openai"
        or any(marker in lowered for marker in _STACK_TRACE_MARKERS)
        or any(token in lowered for token in ("openai_api_key", "serp_api_key", "perplexity_api_key"))
    )
    if suspicious:
        return (
            _GENERIC_IMAGE_ERROR_RECOVERABLE
            if recoverable
            else _GENERIC_IMAGE_ERROR_FATAL
        )
    safe = _sanitize_plain_text(message)
    if not safe:
        return (
            _GENERIC_IMAGE_ERROR_RECOVERABLE
            if recoverable
            else _GENERIC_IMAGE_ERROR_FATAL
        )
    return safe


def _render_preformatted_text(content: str) -> str:
    text = _sanitize_plain_text(content)
    if not text:
        return ""
    return f"<pre>{escape(text)}</pre>"


def _render_list_section(title: str, items: List[str]) -> str:
    safe_items = [_sanitize_plain_text(item) for item in items]
    safe_items = [item for item in safe_items if item]
    if not safe_items:
        return ""
    list_items = "\n".join(f"<li>{escape(item)}</li>" for item in safe_items)
    return f"<section><h2>{escape(title)}</h2><ul>{list_items}</ul></section>"


def _render_workflow_summary(state: Mapping[str, Any]) -> str:
    workflow_status = derive_export_workflow_status(state)
    routing_decision = _sanitize_plain_text(state.get("routing_decision")) or "unknown"
    outputs = [
        _sanitize_plain_text(item).lower()
        for item in _safe_list(state.get("requested_outputs", []))
        if _sanitize_plain_text(item)
    ]
    outputs_text = ", ".join(outputs) if outputs else "none"
    return (
        "<section>"
        "<h2>Workflow Summary</h2>"
        "<ul>"
        f"<li><strong>Workflow Status:</strong> <code>{escape(workflow_status)}</code></li>"
        f"<li><strong>Routing Decision:</strong> <code>{escape(routing_decision)}</code></li>"
        f"<li><strong>Requested Outputs:</strong> {escape(outputs_text)}</li>"
        "</ul>"
        "</section>"
    )


def _render_text_section(title: str, content: str) -> str:
    body = _render_preformatted_text(content)
    if not body:
        return ""
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _render_image_outputs(state: Mapping[str, Any]) -> str:
    entries: List[str] = []
    for raw in _safe_list(state.get("image_outputs", [])):
        item = _safe_dict(raw)
        status = _sanitize_plain_text(item.get("status")).lower() or "unknown"
        provider = _sanitize_plain_text(item.get("provider")) or "unknown"
        url = _safe_href(item.get("url"))
        identifier = _sanitize_plain_text(item.get("id"))

        if status == "failed":
            message = _normalize_image_error_message(item.get("error", {}))
            entries.append(
                f"<li><code>{escape(status)}</code> | <code>{escape(provider)}</code> | {escape(message)}</li>"
            )
            continue

        value = ""
        if url:
            value = f"<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">{url}</a>"
        elif identifier:
            value = escape(identifier)
        else:
            value = "unavailable"

        entries.append(
            f"<li><code>{escape(status)}</code> | <code>{escape(provider)}</code> | {value}</li>"
        )

    if not entries:
        return ""
    return "<section><h2>Image Outputs</h2><ul>" + "".join(entries) + "</ul></section>"


def _render_sources(state: Mapping[str, Any]) -> str:
    citation_checked = validate_citation_sources(
        _safe_list(state.get("sources", [])),
        research_requested=bool(_safe_list(state.get("sources", []))),
    )
    safe_sources = _safe_list(citation_checked.get("sanitized_sources", []))
    deduped: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []
    for idx, raw in enumerate(safe_sources):
        item = _safe_dict(raw)
        url = _safe_href(item.get("url"))
        title = _sanitize_plain_text(item.get("title")) or f"Source {idx + 1}"
        key = f"url:{url}" if url else f"title:{title.lower()}"
        if key not in deduped:
            deduped[key] = item
            ordered_keys.append(key)

    if not ordered_keys:
        return ""

    entries: List[str] = []
    for idx, key in enumerate(ordered_keys, start=1):
        item = _safe_dict(deduped[key])
        title = _sanitize_plain_text(item.get("title")) or f"Source {idx}"
        url = _safe_href(item.get("url"))
        snippet = _sanitize_plain_text(item.get("snippet"))
        citation_available = bool(item.get("citation_available", False)) and bool(url)

        if citation_available:
            title_html = (
                f"<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\">{escape(title)}</a>"
            )
        else:
            title_html = escape(title)

        snippet_html = f"<div class=\"snippet\">{escape(snippet)}</div>" if snippet else ""
        entries.append(f"<li>{title_html}{snippet_html}</li>")

    return "<section><h2>Sources</h2><ol>" + "".join(entries) + "</ol></section>"


def sanitize_html_content(html_text: str) -> str:
    """Strip unsafe executable HTML patterns from rendered output."""
    safe = _safe_text(html_text)
    sanitized, _ = sanitize_html_output(safe)
    return sanitized


def build_html_export_document(state: Mapping[str, Any]) -> str:
    """Build a full safe HTML export document with embedded styling."""
    content_drafts = _safe_dict(state.get("content_drafts", {}))
    blog_body = _safe_text(_safe_dict(content_drafts.get("blog", {})).get("body"))
    linkedin_body = _safe_text(_safe_dict(content_drafts.get("linkedin", {})).get("body"))
    research_body = _safe_text(_safe_dict(content_drafts.get("research_report", {})).get("body"))
    if not research_body:
        research_data = _safe_dict(state.get("research_data", {}))
        research_body = _safe_text(
            research_data.get("synthesized_summary") or research_data.get("summary")
        )

    sections: List[str] = [_render_workflow_summary(state)]

    warnings = collect_export_warnings(state)
    warning_section = _render_list_section("Warnings", warnings)
    if warning_section:
        sections.append(warning_section)

    blog_section = _render_text_section("Blog Draft", blog_body)
    if blog_section:
        sections.append(blog_section)
    linkedin_section = _render_text_section("LinkedIn Draft", linkedin_body)
    if linkedin_section:
        sections.append(linkedin_section)
    research_section = _render_text_section("Research Report", research_body)
    if research_section:
        sections.append(research_section)

    prompts = [
        _safe_text(item)
        for item in _safe_list(state.get("image_prompts", []))
        if _safe_text(item)
    ]
    prompt_section = _render_list_section("Image Prompts", prompts)
    if prompt_section:
        sections.append(prompt_section)

    image_outputs_section = _render_image_outputs(state)
    if image_outputs_section:
        sections.append(image_outputs_section)

    sources_section = _render_sources(state)
    if sources_section:
        sections.append(sources_section)

    body_content = "\n".join(section for section in sections if section)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ContentBlitz Export</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      color: #0f172a;
      background: #f8fbff;
      margin: 0;
      padding: 2rem;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #d7e1ee;
      border-radius: 14px;
      padding: 1.5rem 1.75rem;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    h1 {{
      margin: 0 0 1rem 0;
      font-size: 1.9rem;
    }}
    h2 {{
      margin: 0 0 0.5rem 0;
      font-size: 1.2rem;
    }}
    section {{
      margin-bottom: 1.2rem;
    }}
    ul, ol {{
      margin: 0.35rem 0 0 1.2rem;
      padding: 0;
    }}
    li {{
      margin: 0.35rem 0;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.42;
      background: #f3f7fc;
      border: 1px solid #d7e1ee;
      border-radius: 10px;
      padding: 0.85rem;
    }}
    code {{
      background: #e9eff8;
      border-radius: 5px;
      padding: 0.12rem 0.35rem;
    }}
    a {{
      color: #0b5fa5;
    }}
    .snippet {{
      color: #475569;
      margin-top: 0.2rem;
      font-size: 0.94rem;
    }}
  </style>
</head>
<body>
  <main>
    <h1>ContentBlitz Export</h1>
    {body_content}
  </main>
</body>
</html>
"""
    return sanitize_html_content(document)
