"""Deterministic output sanitization helpers for rendered/exported content."""

from __future__ import annotations

import re
from urllib.parse import urlsplit
from typing import Any, Tuple

_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
    "  file \"",
)
_RAW_PROVIDER_PAYLOAD_MARKERS = (
    "{'code':",
    '"code":',
    "configuration_error",
    "provider':",
    '"provider":',
    "recoverable': false",
    '"recoverable": false',
)
_ENV_NAME_RE = re.compile(
    r"OPENAI_API_KEY|SERP_API_KEY|PERPLEXITY_API_KEY",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\b(?:sk|pplx|serp)_[A-Za-z0-9\-_]{6,}\b|\bsk-[A-Za-z0-9\-_]{4,}\b|\bpplx-[A-Za-z0-9\-_]{6,}\b",
    flags=re.IGNORECASE,
)
_NONE_NULL_RE = re.compile(r"\b(?:none|null)\b", flags=re.IGNORECASE)
_SCRIPT_TAG_RE = re.compile(r"(?is)<script\b[^>]*>.*?</script>")
_UNSAFE_TAG_RE = re.compile(r"(?is)</?(?:iframe|object|embed)\b[^>]*>")
_EVENT_HANDLER_ATTR_RE = re.compile(
    r"""(?is)\s+on[a-z0-9_-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)"""
)
_UNSAFE_URL_SCHEME_RE = re.compile(r"(?i)\b(?:javascript|data|file|ftp|mailto|vbscript)\s*:")
_DATA_IMAGE_RE = re.compile(r"(?i)data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\s]+")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HTML_ATTR_URL_RE = re.compile(
    r"""(?is)\b(href|src)\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def is_safe_external_url(url: str) -> bool:
    candidate = _safe_text(url)
    if not candidate:
        return False
    lowered = candidate.lower()
    if _UNSAFE_URL_SCHEME_RE.search(lowered):
        return False
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    if any(ch in candidate for ch in ("\r", "\n", "\t")):
        return False
    return True


def _remove_unsafe_urls(text: str) -> str:
    def _replace_markdown_image(match: re.Match[str]) -> str:
        alt = _safe_text(match.group(1))
        url = _safe_text(match.group(2))
        if is_safe_external_url(url):
            return f"![{alt}]({url})"
        return ""

    def _replace_markdown_link(match: re.Match[str]) -> str:
        label = _safe_text(match.group(1))
        url = _safe_text(match.group(2))
        if is_safe_external_url(url):
            return f"[{label}]({url})"
        return label

    def _replace_html_attr(match: re.Match[str]) -> str:
        attr = match.group(1).lower()
        value = _safe_text(match.group(2) or match.group(3))
        if is_safe_external_url(value):
            return f'{attr}="{value}"'
        return ""

    sanitized = _MARKDOWN_IMAGE_RE.sub(_replace_markdown_image, text)
    sanitized = _MARKDOWN_LINK_RE.sub(_replace_markdown_link, sanitized)
    sanitized = _HTML_ATTR_URL_RE.sub(_replace_html_attr, sanitized)
    sanitized = _UNSAFE_URL_SCHEME_RE.sub("", sanitized)
    return sanitized


def _strip_unsafe_markup(text: str) -> str:
    sanitized = _SCRIPT_TAG_RE.sub("", text)
    sanitized = _UNSAFE_TAG_RE.sub("", sanitized)
    sanitized = _EVENT_HANDLER_ATTR_RE.sub("", sanitized)
    sanitized = _DATA_IMAGE_RE.sub("", sanitized)
    return sanitized


def _redact_sensitive_tokens(text: str) -> str:
    sanitized = _ENV_NAME_RE.sub("[REDACTED]", text)
    sanitized = _TOKEN_RE.sub("[REDACTED]", sanitized)
    sanitized = _CONTROL_CHARS_RE.sub("", sanitized)
    return sanitized


def _strip_unsafe_lines(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if any(marker in lowered for marker in _STACK_TRACE_MARKERS):
            continue
        if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
            continue
        if "data:image/" in lowered or "b64_json" in lowered or "base64" in lowered:
            continue
        cleaned = _NONE_NULL_RE.sub("", line).rstrip()
        if cleaned:
            kept.append(cleaned)
        elif kept and kept[-1]:
            kept.append("")

    while kept and not kept[0]:
        kept.pop(0)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept).strip()


def sanitize_markdown_output(value: Any) -> Tuple[str, bool]:
    raw = _safe_text(value)
    if not raw:
        return "", False
    sanitized = _strip_unsafe_markup(raw)
    sanitized = _remove_unsafe_urls(sanitized)
    sanitized = _redact_sensitive_tokens(sanitized)
    sanitized = _strip_unsafe_lines(sanitized)
    return sanitized, sanitized != raw


def sanitize_html_output(value: Any) -> Tuple[str, bool]:
    raw = _safe_text(value)
    if not raw:
        return "", False
    sanitized = _strip_unsafe_markup(raw)
    sanitized = _remove_unsafe_urls(sanitized)
    sanitized = _redact_sensitive_tokens(sanitized)
    sanitized = _strip_unsafe_lines(sanitized)
    return sanitized, sanitized != raw


def sanitize_plain_output(value: Any) -> Tuple[str, bool]:
    raw = _safe_text(value)
    if not raw:
        return "", False
    sanitized = _strip_unsafe_markup(raw)
    sanitized = _remove_unsafe_urls(sanitized)
    sanitized = _redact_sensitive_tokens(sanitized)
    sanitized = _strip_unsafe_lines(sanitized)
    return sanitized, sanitized != raw
