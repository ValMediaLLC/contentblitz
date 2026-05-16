"""Safe, user-facing error normalization for UI rendering."""

from __future__ import annotations

import re
from typing import Any, Mapping

_MAX_RENDERED_ERROR_LENGTH = 500
_GENERIC_FATAL_MESSAGE = "The workflow ended due to an internal error."
_GENERIC_RECOVERABLE_MESSAGE = "A recoverable workflow error occurred."

_SECRET_ASSIGNMENT_PATTERNS = (
    re.compile(r"(OPENAI_API_KEY\s*[:=]\s*)([^\s,;]+)", flags=re.IGNORECASE),
    re.compile(r"(SERP_API_KEY\s*[:=]\s*)([^\s,;]+)", flags=re.IGNORECASE),
    re.compile(r"(PERPLEXITY_API_KEY\s*[:=]\s*)([^\s,;]+)", flags=re.IGNORECASE),
)
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9\-_]{8,}\b"),
    re.compile(r"\bpplx-[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
    re.compile(r"\bserp_[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
)
_TRACEBACK_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _contains_stack_trace(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _TRACEBACK_MARKERS):
        return True
    if '  file "' in lowered and " line " in lowered:
        return True
    return False


def redact_sensitive_text(text: str) -> str:
    """Redact credential-like substrings from arbitrary text."""
    redacted = _safe_text(text)
    if not redacted:
        return ""

    for pattern in _SECRET_ASSIGNMENT_PATTERNS:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _safe_message(
    raw_message: str,
    *,
    recoverable: bool,
    default_message: str | None = None,
) -> str:
    base_default = (
        _GENERIC_RECOVERABLE_MESSAGE if recoverable else _GENERIC_FATAL_MESSAGE
    )
    fallback = _safe_text(default_message) or base_default
    candidate = _safe_text(raw_message) or fallback
    if candidate.strip().lower() in {"none", "null"}:
        candidate = fallback

    if _contains_stack_trace(candidate):
        candidate = fallback

    candidate = redact_sensitive_text(candidate)
    if len(candidate) > _MAX_RENDERED_ERROR_LENGTH:
        candidate = candidate[:_MAX_RENDERED_ERROR_LENGTH].rstrip() + "..."
    return candidate


def normalize_error_for_display(
    error: Any,
    *,
    default_message: str | None = None,
) -> dict[str, Any]:
    """
    Normalize arbitrary error payloads into a user-safe display contract.

    Returned fields intentionally avoid raw provider payloads/stack traces.
    """
    code = "unknown_error"
    recoverable = False
    agent = ""
    provider = ""
    raw_message = ""

    if isinstance(error, Mapping):
        code = _safe_text(error.get("code") or error.get("type") or "unknown_error")
        recoverable = bool(error.get("recoverable", False))
        agent = _safe_text(error.get("agent"))
        provider = _safe_text(error.get("provider"))
        raw_message = _safe_text(error.get("message"))
    else:
        raw_message = _safe_text(error)

    safe_message = _safe_message(
        raw_message,
        recoverable=recoverable,
        default_message=default_message,
    )

    normalized: dict[str, Any] = {
        "code": code or "unknown_error",
        "message": safe_message,
        "recoverable": recoverable,
    }
    if agent:
        normalized["agent"] = agent
    if provider:
        normalized["provider"] = provider
    return normalized


def normalize_errors_for_display(errors: Any) -> list[dict[str, Any]]:
    """Normalize an error list into user-safe error dicts."""
    if not isinstance(errors, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in errors:
        normalized.append(normalize_error_for_display(item))
    return normalized
