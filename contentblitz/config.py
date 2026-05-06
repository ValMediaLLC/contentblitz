"""Configuration defaults for the ContentBlitz Phase 1 scaffold."""

from __future__ import annotations

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
    "primary_image_provider": "dall-e-3",
    "fallback_image_provider": "dall-e-2",
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


def validate_retry_policy_keys(retry_counts: Mapping[str, int]) -> bool:
    """Return True when retry policy keys match state retry count keys exactly."""
    return set(RETRY_POLICY.keys()) == set(retry_counts.keys())


def build_cache_metadata_defaults() -> Dict[str, Any]:
    """Return a deep copy of cache metadata defaults."""
    return deepcopy(CACHE_METADATA_DEFAULTS)


def build_cost_controls_defaults() -> Dict[str, Any]:
    """Return a deep copy of cost control defaults."""
    return deepcopy(COST_CONTROLS_DEFAULTS)

