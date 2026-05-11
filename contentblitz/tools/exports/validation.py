"""Validation helpers for markdown export safety and structure."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
    "  file \"",
)
_RAW_PROVIDER_PAYLOAD_MARKERS = (
    "{'code':",
    '"code":',
    "configuration_error",
    "provider': 'openai'",
    '"provider": "openai"',
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


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _contains_stack_trace(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STACK_TRACE_MARKERS)


def validate_markdown_export(
    markdown_text: str,
    *,
    sources_exist: bool = False,
) -> Dict[str, Any]:
    """
    Validate markdown export payload for structure and safety.

    Returns structured validation metadata:
    {"valid": bool, "warnings": [...], "errors": [...]}
    """
    text = _safe_text(markdown_text)
    warnings: List[str] = []
    errors: List[str] = []

    if not text:
        errors.append("Export content is empty.")
        return {"valid": False, "warnings": warnings, "errors": errors}

    lines = text.splitlines()
    first_non_empty = next((line.strip() for line in lines if line.strip()), "")
    if not first_non_empty.startswith("# "):
        errors.append("Missing required top-level markdown heading.")

    lowered = text.lower()
    if _NONE_NULL_RE.search(text):
        warnings.append("Found null-like placeholder text.")
    if _contains_stack_trace(text):
        errors.append("Stack trace content is not allowed in exports.")
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        errors.append("Base64 image payload is not allowed in exports.")
    if any(env_name in lowered for env_name in _ENV_NAME_PATTERNS):
        errors.append("Environment variable names are not allowed in exports.")
    if any(pattern.search(text) for pattern in _TOKEN_PATTERNS):
        errors.append("Credential-like token content is not allowed in exports.")
    if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
        errors.append("Raw provider/configuration payload content is not allowed in exports.")
    if sources_exist and "## sources" not in lowered:
        errors.append("Sources section is required when sources are present.")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def normalize_validation_result(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize arbitrary validation payload into a strict safe shape."""
    valid = bool(payload.get("valid", False))
    warnings = [
        _safe_text(item) for item in payload.get("warnings", []) if _safe_text(item)
    ] if isinstance(payload.get("warnings", []), list) else []
    errors = [
        _safe_text(item) for item in payload.get("errors", []) if _safe_text(item)
    ] if isinstance(payload.get("errors", []), list) else []
    return {"valid": valid, "warnings": warnings, "errors": errors}
