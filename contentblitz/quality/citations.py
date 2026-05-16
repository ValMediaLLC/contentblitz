"""Deterministic citation validation helpers."""

from __future__ import annotations

import re
from urllib.parse import urlsplit
from typing import Any, Dict, List, Mapping

_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
    '  file "',
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
_ENV_NAME_PATTERNS = (
    "openai_api_key",
    "serp_api_key",
    "perplexity_api_key",
)
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9\-_]{8,}\b"),
    re.compile(r"\bpplx-[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
    re.compile(r"\bserp_[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
)
_NONE_NULL_RE = re.compile(r"\b(?:none|null)\b", flags=re.IGNORECASE)

CITATION_VALIDATION_WARNING = (
    "Citation validation found missing, duplicate, or unsafe source entries."
)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_safe_url(url: str) -> bool:
    candidate = _safe_text(url)
    if not candidate:
        return False
    lowered = candidate.lower()
    if lowered.startswith(("javascript:", "data:", "file:", "ftp:", "mailto:")):
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


def _has_unsafe_text(value: str) -> bool:
    lowered = _safe_text(value).lower()
    if not lowered:
        return False
    if _NONE_NULL_RE.search(lowered):
        return True
    if any(marker in lowered for marker in _STACK_TRACE_MARKERS):
        return True
    if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
        return True
    if any(pattern.search(lowered) for pattern in _TOKEN_PATTERNS):
        return True
    if any(name in lowered for name in _ENV_NAME_PATTERNS):
        return True
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        return True
    if lowered.startswith("{") and lowered.endswith("}") and ":" in lowered:
        return True
    if lowered.startswith("[") and lowered.endswith("]") and ":" in lowered:
        return True
    return False


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def validate_citation_sources(
    sources: Any,
    *,
    research_requested: bool = False,
) -> Dict[str, Any]:
    """
    Validate and sanitize citation/source records.

    Returns:
    {
      "status": "passed" | "degraded",
      "sanitized_sources": [...],
      "invalid_count": int,
      "duplicate_count": int,
      "unsafe_url_count": int,
      "missing_count": int,
      "warning": str,
      "valid_source_count": int,
    }
    """
    invalid_count = 0
    duplicate_count = 0
    unsafe_url_count = 0
    missing_count = 0

    seen_urls: set[str] = set()
    seen_title_url: set[str] = set()
    sanitized_sources: List[Dict[str, Any]] = []

    for raw in _safe_list(sources):
        if not isinstance(raw, Mapping):
            invalid_count += 1
            continue

        source = _safe_dict(raw)
        title = _safe_text(source.get("title"))
        snippet = _safe_text(source.get("snippet"))
        raw_url = _safe_text(source.get("url"))

        if not title or _has_unsafe_text(title):
            invalid_count += 1
            missing_count += 1 if not title else 0
            continue
        if not snippet or _has_unsafe_text(snippet):
            invalid_count += 1
            missing_count += 1 if not snippet else 0
            continue

        safe_url = ""
        if raw_url:
            if _is_safe_url(raw_url):
                safe_url = raw_url
            else:
                invalid_count += 1
                unsafe_url_count += 1

        if safe_url:
            url_key = safe_url.lower()
            if url_key in seen_urls:
                invalid_count += 1
                duplicate_count += 1
                continue
            seen_urls.add(url_key)

            pair_key = f"{title.lower()}|{url_key}"
            if pair_key in seen_title_url:
                invalid_count += 1
                duplicate_count += 1
                continue
            seen_title_url.add(pair_key)

        sanitized_sources.append(
            {
                "title": title,
                "url": safe_url or None,
                "snippet": snippet,
                "source": _safe_text(source.get("source")),
                "published_at": _safe_text(source.get("published_at")) or None,
                "citation_available": bool(safe_url),
                "credibility_score": _as_float(
                    source.get("credibility_score"), default=0.0
                ),
            }
        )

    valid_source_count = len(sanitized_sources)
    needs_warning = (
        invalid_count > 0
        or duplicate_count > 0
        or unsafe_url_count > 0
        or (research_requested and valid_source_count == 0)
    )
    status = "degraded" if needs_warning else "passed"
    warning = CITATION_VALIDATION_WARNING if needs_warning else ""
    return {
        "status": status,
        "sanitized_sources": sanitized_sources,
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "unsafe_url_count": unsafe_url_count,
        "missing_count": missing_count,
        "warning": warning,
        "valid_source_count": valid_source_count,
    }
