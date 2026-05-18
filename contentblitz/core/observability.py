"""Observability configuration helpers for optional LangSmith tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from contentblitz.config import (
    langsmith_api_key_present,
    langsmith_endpoint,
    langsmith_project,
    langsmith_tracing_requested,
)


@dataclass(frozen=True)
class ObservabilityConfig:
    """Public, secret-safe observability configuration."""

    tracing_requested: bool
    tracing_enabled: bool
    endpoint: str
    project: str
    status: str
    message: str


def build_observability_config() -> ObservabilityConfig:
    """Build secret-safe, import-time-safe observability settings."""
    tracing_requested = langsmith_tracing_requested()
    has_api_key = langsmith_api_key_present()
    tracing_enabled = bool(tracing_requested and has_api_key)

    if tracing_enabled:
        status = "enabled"
        message = "LangSmith tracing is enabled."
    elif tracing_requested:
        status = "degraded"
        message = (
            "LangSmith tracing was requested but LANGSMITH_API_KEY is missing. "
            "Tracing remains disabled."
        )
    else:
        status = "disabled"
        message = "LangSmith tracing is disabled."

    return ObservabilityConfig(
        tracing_requested=tracing_requested,
        tracing_enabled=tracing_enabled,
        endpoint=langsmith_endpoint(),
        project=langsmith_project(),
        status=status,
        message=message,
    )


def is_tracing_enabled() -> bool:
    """Return whether tracing is currently active."""
    return build_observability_config().tracing_enabled


def observability_summary() -> Dict[str, str | bool]:
    """Return a secret-safe observability snapshot for UI/logging/debug."""
    config = build_observability_config()
    return {
        "tracing_requested": config.tracing_requested,
        "tracing_enabled": config.tracing_enabled,
        "endpoint": config.endpoint,
        "project": config.project,
        "status": config.status,
        "message": config.message,
    }
