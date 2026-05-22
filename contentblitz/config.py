"""Configuration defaults for the ContentBlitz Phase 1 scaffold."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Mapping

RETRY_POLICY: Dict[str, int] = {
    "query_handler": 1,
    "research_agent": 1,
    "content_strategist": 1,
    "blog_writer": 1,
    "linkedin_writer": 1,
    "image_agent": 1,
    "quality_validator": 1,
    "output_assembler": 1,
    "export": 1,
}

CACHE_METADATA_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "ttl_seconds": 1800,
    "backend": "in_memory",
    "keys": [],
}

COST_CONTROLS_DEFAULTS: Dict[str, Any] = {
    "tokens_used_this_session": 0,
    "search_queries_used_this_session": 0,
    "image_generations_used_this_session": 0,
    "total_retries_used_this_session": 0,
    "budget_exceeded": False,
}

MODEL_FALLBACKS: Dict[str, str] = {
    "primary_text_model": "gpt-4o",
    "fallback_text_model": "gpt-4o-mini",
    "primary_search_provider": "serp_api",
    "fallback_search_provider": "perplexity",
    "primary_image_provider": "stability_ai",
    "fallback_image_provider": "fal_ai",
    "primary_image_model": "stable-image-core",
    "fallback_image_model": "fal-ai/flux/schnell",
}

INJECTION_GUARD = {
    "block_system_override": True,
    "block_external_instructions": True,
    "sanitize_user_input": True,
    "sanitize_research_data": True,
    "strip_code_execution": True,
    "max_input_length": 20000,
    "blocked_patterns": [
        "ignore previous instructions",
        "system override",
        "execute this code",
        "<script>",
        "rm -rf",
        "import os",
    ],
}

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}
LANGSMITH_ENDPOINT_DEFAULT = "https://api.smith.langchain.com"
LANGSMITH_PROJECT_DEFAULT = "ContentBlitz"


def _read_bool_env(var_name: str, *, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if token in _TRUE_ENV_VALUES:
        return True
    if token in _FALSE_ENV_VALUES:
        return False
    return default


def _read_text_env(var_name: str, *, default: str) -> str:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value or default


def live_provider_calls_enabled() -> bool:
    """Return whether live provider API calls are enabled."""
    return _read_bool_env("CONTENTBLITZ_ENABLE_LIVE_CALLS", default=True)


def validate_retry_policy_keys(retry_counts: Mapping[str, int]) -> bool:
    """Return True when retry policy keys match state retry count keys exactly."""
    return set(RETRY_POLICY.keys()) == set(retry_counts.keys())


def build_cache_metadata_defaults() -> Dict[str, Any]:
    """Return a deep copy of cache metadata defaults."""
    defaults = deepcopy(CACHE_METADATA_DEFAULTS)
    raw_ttl = os.getenv("CONTENTBLITZ_CACHE_TTL_SECONDS")
    if raw_ttl is not None:
        try:
            defaults["ttl_seconds"] = max(0, int(raw_ttl.strip()))
        except (TypeError, ValueError):
            defaults["ttl_seconds"] = CACHE_METADATA_DEFAULTS["ttl_seconds"]

    raw_backend = (
        str(os.getenv("CONTENTBLITZ_CACHE_BACKEND", defaults["backend"]))
        .strip()
        .lower()
    )
    if raw_backend in {"sqlite", "in_memory", "in-memory", "memory", "inmemory"}:
        defaults["backend"] = "sqlite" if raw_backend == "sqlite" else "in_memory"
    else:
        defaults["backend"] = CACHE_METADATA_DEFAULTS["backend"]
    return defaults


def build_cost_controls_defaults() -> Dict[str, Any]:
    """Return a deep copy of cost control defaults."""
    return deepcopy(COST_CONTROLS_DEFAULTS)


def langsmith_tracing_requested() -> bool:
    """Return whether LangSmith tracing was explicitly requested."""
    return _read_bool_env("LANGSMITH_TRACING", default=False)


def langsmith_api_key_present() -> bool:
    """Return whether a non-empty LangSmith API key is present."""
    raw = os.getenv("LANGSMITH_API_KEY")
    if raw is None:
        return False
    return bool(str(raw).strip())


def langsmith_endpoint() -> str:
    """Return LangSmith endpoint or a safe cloud default."""
    return _read_text_env("LANGSMITH_ENDPOINT", default=LANGSMITH_ENDPOINT_DEFAULT)


def langsmith_project() -> str:
    """Return LangSmith project name or a safe default."""
    return _read_text_env("LANGSMITH_PROJECT", default=LANGSMITH_PROJECT_DEFAULT)
