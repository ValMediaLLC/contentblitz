"""Shared user-facing warning normalization helpers."""

from __future__ import annotations

from typing import Any, Iterable

TOP_LEVEL_PROVIDER_WARNING = (
    "Provider unavailable or quota-limited. "
    "ContentBlitz generated limited fallback outputs."
)
TEXT_FALLBACK_WARNING = (
    "Text generation was unavailable. "
    "Blog and LinkedIn outputs are fallback outlines."
)
IMAGE_RECOVERABLE_WARNING = (
    "Image generation encountered a recoverable issue. "
    "Text/research/export outputs remain available."
)

_TEXT_WARNING_MARKERS = (
    "draft unavailable because text generation is currently limited",
    "text generation provider was unavailable or quota-limited",
    "text generation was unavailable",
    "fallback draft content is limited",
    "fallback outlines",
)
_TOP_LEVEL_PROVIDER_MARKERS = (
    "provider unavailable or quota-limited",
    "text provider unavailable or quota-limited",
    "openai provider unavailable or quota-limited",
    "anthropic text provider unavailable or quota-limited",
    (
        "provider unavailable or quota-limited. "
        "contentblitz generated limited fallback outputs"
    ),
)
_IMAGE_WARNING_MARKERS = (
    "image generation encountered a recoverable issue",
    "image generation failed in this run",
    "image generation is temporarily unavailable",
)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_user_warning(message: Any) -> str:
    """Normalize semantically-duplicate warnings into canonical user-safe text."""
    cleaned = _safe_text(message)
    if not cleaned:
        return ""
    lowered = cleaned.lower()

    if any(marker in lowered for marker in _TOP_LEVEL_PROVIDER_MARKERS):
        return TOP_LEVEL_PROVIDER_WARNING
    if any(marker in lowered for marker in _TEXT_WARNING_MARKERS):
        return TEXT_FALLBACK_WARNING
    if any(marker in lowered for marker in _IMAGE_WARNING_MARKERS):
        return IMAGE_RECOVERABLE_WARNING
    return cleaned


def dedupe_user_warnings(messages: Iterable[Any]) -> list[str]:
    """Deduplicate warnings while preserving first occurrence order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = normalize_user_warning(message)
        if not normalized:
            continue
        key = normalized.lower()
        if key in {"none", "null"} or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped
