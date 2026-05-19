"""Trace-safe redaction and metadata sanitization helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

REDACTED_TOKEN = "[REDACTED]"
REDACTED_STACK_TRACE = "[REDACTED_STACK_TRACE]"
REDACTED_BASE64_PAYLOAD = "[REDACTED_BASE64_PAYLOAD]"
REDACTED_RAW_PAYLOAD = "[REDACTED_RAW_PAYLOAD]"
TRUNCATED_SUFFIX = "... [TRUNCATED]"
TRUNCATED_ITEMS_TOKEN = "[TRUNCATED_ITEMS]"
TRUNCATED_DEPTH_TOKEN = "[TRUNCATED_DEPTH]"

MAX_TRACE_STRING_LENGTH = 280
MAX_TRACE_PREVIEW_CHARS = 96
MAX_TRACE_LIST_ITEMS = 24
MAX_TRACE_DICT_ITEMS = 40
MAX_TRACE_NESTED_DEPTH = 6
MAX_ERROR_MESSAGE_LENGTH = 220

_KNOWN_ENV_NAME_RE = re.compile(
    r"OPENAI_API_KEY|LANGSMITH_API_KEY|SERP_API_KEY|PERPLEXITY_API_KEY",
    flags=re.IGNORECASE,
)
_ENV_ASSIGNMENT_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]{2,}(?:API_KEY|TOKEN|SECRET|PASSWORD|PASS|KEY))\s*[:=]\s*([^\s,;]+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+([A-Za-z0-9\-._~+/=]{8,})")
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9\-_]{4,}\b", flags=re.IGNORECASE),
    re.compile(r"\blsv2_[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
    re.compile(r"\bpplx-[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
    re.compile(r"\bserp_[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        flags=re.IGNORECASE,
    ),
)
_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
)
_PY_STACK_FRAME_RE = re.compile(
    r'(?im)^\s*File\s+"[^"]+",\s+line\s+\d+(?:,\s+in\s+[^\n]+)?\s*$'
)
_PY_EXCEPTION_RE = re.compile(
    r"(?im)^\s*[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)\s*:\s+.+$"
)
_JS_STACK_FRAME_RE = re.compile(
    r"(?im)^\s*at\s+.+\((?:[a-z]:)?[^:]+:\d+:\d+\)\s*$"
)
_DATA_IMAGE_RE = re.compile(
    r"(?is)data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\s]+"
)
_BASE64_BLOCK_RE = re.compile(
    r"\b(?=[A-Za-z0-9+/]{80,}={0,2}\b)"
    r"(?=[A-Za-z0-9+/=]*[0-9+/=])"
    r"[A-Za-z0-9+/]{80,}={0,2}\b"
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_RAW_PAYLOAD_KEY_RE = re.compile(
    r"(?:^|_)(raw|payload|request|headers|body|full_text)(?:$|_)",
    flags=re.IGNORECASE,
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(text: str, *, max_length: int = MAX_TRACE_STRING_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + TRUNCATED_SUFFIX


def _safe_word_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len([token for token in stripped.split() if token.strip()])


def _safe_line_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.splitlines())


def _contains_stack_trace(text: str) -> bool:
    lowered = text.lower()
    if "traceback (most recent call last):" in lowered:
        return True

    has_py_frame = _PY_STACK_FRAME_RE.search(text) is not None
    has_py_exception = _PY_EXCEPTION_RE.search(text) is not None
    if has_py_frame and has_py_exception:
        return True

    has_js_frame = _JS_STACK_FRAME_RE.search(text) is not None
    if has_js_frame and ("error" in lowered or "exception" in lowered):
        return True

    if "stack trace" in lowered and (has_py_frame or has_js_frame):
        return True
    return False


def _looks_like_raw_payload_key(key: str) -> bool:
    candidate = _safe_text(key)
    if not candidate:
        return False
    return _RAW_PAYLOAD_KEY_RE.search(candidate) is not None


def safe_text_preview(
    value: Any,
    *,
    max_preview_chars: int = MAX_TRACE_PREVIEW_CHARS,
) -> str:
    """Return a short, redacted preview suitable for trace metadata."""
    text = redact_sensitive_text(str(value or ""), max_length=max_preview_chars)
    return text


def summarize_text_content(
    value: Any,
    *,
    max_preview_chars: int = MAX_TRACE_PREVIEW_CHARS,
) -> dict[str, Any]:
    """Return compact summary metadata for potentially large text content."""
    raw_text = str(value or "")
    preview = safe_text_preview(raw_text, max_preview_chars=max_preview_chars)
    digest = hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return {
        "length": len(raw_text),
        "word_count": _safe_word_count(raw_text),
        "line_count": _safe_line_count(raw_text),
        "preview": preview,
        "sha256_prefix": digest,
    }


def redact_sensitive_text(
    text: str,
    *,
    max_length: int = MAX_TRACE_STRING_LENGTH,
) -> str:
    """Redact secrets and unsafe payloads from arbitrary text."""
    candidate = _safe_text(text)
    if not candidate:
        return ""

    if _contains_stack_trace(candidate):
        return REDACTED_STACK_TRACE

    redacted = _DATA_IMAGE_RE.sub(REDACTED_BASE64_PAYLOAD, candidate)
    redacted = _BASE64_BLOCK_RE.sub(REDACTED_BASE64_PAYLOAD, redacted)
    redacted = _ENV_ASSIGNMENT_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _KNOWN_ENV_NAME_RE.sub(REDACTED_TOKEN, redacted)
    redacted = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", redacted)
    for pattern in _TOKEN_PATTERNS:
        redacted = pattern.sub(REDACTED_TOKEN, redacted)
    redacted = _CONTROL_CHARS_RE.sub("", redacted)
    return _truncate_text(redacted, max_length=max_length)


def sanitize_trace_value(
    value: Any,
    *,
    max_string_length: int = MAX_TRACE_STRING_LENGTH,
    _depth: int = 0,
) -> Any:
    """
    Recursively sanitize a value so it is safe and JSON-serializable.

    This helper never mutates input objects.
    """
    if _depth >= MAX_TRACE_NESTED_DEPTH:
        return TRUNCATED_DEPTH_TOKEN
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return redact_sensitive_text(value, max_length=max_string_length)
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_TRACE_DICT_ITEMS:
                sanitized["_truncated"] = True
                break
            safe_key = redact_sensitive_text(str(key), max_length=64) or "key"
            if _looks_like_raw_payload_key(safe_key):
                sanitized[safe_key] = REDACTED_RAW_PAYLOAD
            else:
                sanitized[safe_key] = sanitize_trace_value(
                    item,
                    max_string_length=max_string_length,
                    _depth=_depth + 1,
                )
        return sanitized
    if isinstance(value, (list, tuple, set)):
        sanitized_items: list[Any] = []
        for index, item in enumerate(value):
            if index >= MAX_TRACE_LIST_ITEMS:
                sanitized_items.append(TRUNCATED_ITEMS_TOKEN)
                break
            sanitized_items.append(
                sanitize_trace_value(
                    item,
                    max_string_length=max_string_length,
                    _depth=_depth + 1,
                )
            )
        return sanitized_items
    return redact_sensitive_text(str(value), max_length=max_string_length)


def normalize_trace_error(error: Any) -> dict[str, Any]:
    """Normalize raw exceptions/error payloads into a safe trace error summary."""
    recoverable = False
    code = "unknown_error"
    raw_message = ""

    if isinstance(error, BaseException):
        raw_message = f"{error.__class__.__name__}: {_safe_text(error)}"
    elif isinstance(error, Mapping):
        recoverable = bool(error.get("recoverable", False))
        code = _safe_text(error.get("code") or error.get("type") or code) or code
        raw_message = _safe_text(error.get("message"))
    else:
        raw_message = _safe_text(error)

    if _contains_stack_trace(raw_message):
        safe_message = (
            "A recoverable provider error occurred."
            if recoverable
            else "A workflow error occurred."
        )
    else:
        safe_message = redact_sensitive_text(
            raw_message,
            max_length=MAX_ERROR_MESSAGE_LENGTH,
        )

    if not safe_message:
        safe_message = (
            "A recoverable provider error occurred."
            if recoverable
            else "A workflow error occurred."
        )

    return {
        "code": redact_sensitive_text(code, max_length=64) or "unknown_error",
        "recoverable": recoverable,
        "message": safe_message,
    }
