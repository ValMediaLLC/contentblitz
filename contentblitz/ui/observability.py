"""UI-safe observability diagnostics helpers."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from contentblitz.core.observability import (
    ObservabilityConfig,
    build_observability_config,
)
from contentblitz.ui.status import (
    normalize_observability_status,
    observability_status_label,
)

_STATUS_NOTE = {
    "enabled": "Tracing is enabled for this app session.",
    "disabled": "Tracing is disabled. Workflow execution continues without tracing.",
    "degraded": "Tracing is unavailable. Workflow execution continues without tracing.",
}
_STATUS_TONE = {
    "enabled": "cbx-status-green",
    "disabled": "cbx-status-orange",
    "degraded": "cbx-status-red",
}
_TRACE_ATTEMPT_LABELS = {
    "ready": "Ready",
    "not_requested": "Not requested",
    "unavailable": "Unavailable",
}
_TRACE_ATTEMPT_FALLBACK = {
    "enabled": "ready",
    "disabled": "not_requested",
    "degraded": "unavailable",
}
_DASHBOARD_INSTRUCTION = (
    "For trace details, review the LangSmith dashboard manually."
)


def _safe_text(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return ""
    return text


def _safe_endpoint_host(endpoint: str) -> str:
    candidate = _safe_text(endpoint)
    if not candidate:
        return "unknown"
    parsed = urlparse(candidate)
    host = _safe_text(parsed.hostname)
    if host:
        return host
    if "://" not in candidate:
        fallback_parsed = urlparse(f"https://{candidate}")
        fallback_host = _safe_text(fallback_parsed.hostname)
        if fallback_host:
            return fallback_host
    return "unknown"


def _safe_trace_attempt_status(
    *,
    status: str,
    last_trace_attempt_status: str = "",
) -> str:
    normalized = _safe_text(last_trace_attempt_status).lower()
    if normalized in _TRACE_ATTEMPT_LABELS:
        return normalized
    return _TRACE_ATTEMPT_FALLBACK.get(status, "not_requested")


def build_observability_diagnostics(
    *,
    config: ObservabilityConfig | None = None,
    last_trace_attempt_status: str = "",
) -> dict[str, Any]:
    """
    Build a secret-safe observability diagnostics payload for frontend rendering.

    This helper never calls providers or LangSmith APIs and never returns secrets.
    """
    resolved_config = config or build_observability_config()
    status = normalize_observability_status(resolved_config.status)
    project_name = _safe_text(resolved_config.project) or "ContentBlitz"
    endpoint_host = _safe_endpoint_host(_safe_text(resolved_config.endpoint))
    trace_attempt_status = _safe_trace_attempt_status(
        status=status,
        last_trace_attempt_status=last_trace_attempt_status,
    )
    return {
        "status": status,
        "status_label": observability_status_label(status),
        "status_tone_class": _STATUS_TONE.get(status, "cbx-status-orange"),
        "tracing_enabled": bool(resolved_config.tracing_enabled),
        "project_name": project_name,
        "endpoint_host": endpoint_host,
        "last_trace_attempt_status": trace_attempt_status,
        "last_trace_attempt_label": _TRACE_ATTEMPT_LABELS[trace_attempt_status],
        "note": _STATUS_NOTE.get(status, _STATUS_NOTE["disabled"]),
        "dashboard_instruction": _DASHBOARD_INSTRUCTION,
    }
