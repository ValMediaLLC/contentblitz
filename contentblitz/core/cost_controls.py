"""Deterministic cost-control helpers for agent-owned counter updates."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, Mapping

from contentblitz.config import COST_CONTROLS_DEFAULTS

DEFAULT_TOKEN_BUDGET_PER_SESSION = 10000
DEFAULT_SEARCH_QUERY_CAP_PER_SESSION = 5
DEFAULT_IMAGE_GENERATION_CAP_PER_SESSION = 3
DEFAULT_MAX_TOTAL_RETRIES_PER_SESSION = 3
NEAR_TOKEN_BUDGET_RATIO = 0.90

PRIMARY_TEXT_MODEL = "gpt-4o"
FALLBACK_TEXT_MODEL = "gpt-4o-mini"


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def normalize_cost_controls(raw: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return a normalized mutable cost-controls dict with deterministic defaults."""
    base = deepcopy(COST_CONTROLS_DEFAULTS)
    incoming = dict(raw) if isinstance(raw, Mapping) else {}
    base.update(incoming)

    base["tokens_used_this_session"] = max(0, _safe_int(base.get("tokens_used_this_session", 0), 0))
    base["search_queries_used_this_session"] = max(
        0,
        _safe_int(base.get("search_queries_used_this_session", 0), 0),
    )
    base["image_generations_used_this_session"] = max(
        0,
        _safe_int(base.get("image_generations_used_this_session", 0), 0),
    )
    base["total_retries_used_this_session"] = max(
        0,
        _safe_int(base.get("total_retries_used_this_session", 0), 0),
    )
    base["token_budget_per_session"] = max(
        0,
        _safe_int(base.get("token_budget_per_session", DEFAULT_TOKEN_BUDGET_PER_SESSION), DEFAULT_TOKEN_BUDGET_PER_SESSION),
    )
    base["search_query_cap_per_session"] = max(
        0,
        _safe_int(
            base.get("search_query_cap_per_session", DEFAULT_SEARCH_QUERY_CAP_PER_SESSION),
            DEFAULT_SEARCH_QUERY_CAP_PER_SESSION,
        ),
    )
    base["image_generation_cap_per_session"] = max(
        0,
        _safe_int(
            base.get("image_generation_cap_per_session", DEFAULT_IMAGE_GENERATION_CAP_PER_SESSION),
            DEFAULT_IMAGE_GENERATION_CAP_PER_SESSION,
        ),
    )
    base["max_total_retries_per_session"] = max(
        0,
        _safe_int(
            base.get("max_total_retries_per_session", DEFAULT_MAX_TOTAL_RETRIES_PER_SESSION),
            DEFAULT_MAX_TOTAL_RETRIES_PER_SESSION,
        ),
    )
    base["budget_exceeded"] = bool(base.get("budget_exceeded", False))

    if token_budget_exceeded(base):
        base["budget_exceeded"] = True
    return base


def extract_total_tokens_from_text_response(response: Mapping[str, Any] | None) -> int:
    """Extract total token usage from legacy text-tool dict responses."""
    if not isinstance(response, Mapping):
        return 0

    usage = response.get("usage", {})
    if isinstance(usage, Mapping):
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, (int, float)):
            return max(0, int(total_tokens))

    for key in ("tokens_used", "total_tokens", "token_count"):
        value = response.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value))

    metadata = response.get("metadata", {})
    if isinstance(metadata, Mapping):
        meta_tokens = metadata.get("tokens_used")
        if isinstance(meta_tokens, (int, float)):
            return max(0, int(meta_tokens))
    return 0


def token_budget_exceeded(cost_controls: Mapping[str, Any]) -> bool:
    budget = max(0, _safe_int(cost_controls.get("token_budget_per_session", DEFAULT_TOKEN_BUDGET_PER_SESSION), DEFAULT_TOKEN_BUDGET_PER_SESSION))
    used = max(0, _safe_int(cost_controls.get("tokens_used_this_session", 0), 0))
    if budget <= 0:
        return False
    return used >= budget


def near_token_budget(cost_controls: Mapping[str, Any]) -> bool:
    budget = max(0, _safe_int(cost_controls.get("token_budget_per_session", DEFAULT_TOKEN_BUDGET_PER_SESSION), DEFAULT_TOKEN_BUDGET_PER_SESSION))
    used = max(0, _safe_int(cost_controls.get("tokens_used_this_session", 0), 0))
    if budget <= 0:
        return False
    threshold = int(math.floor(budget * NEAR_TOKEN_BUDGET_RATIO))
    return used >= threshold


def preferred_text_model(cost_controls: Mapping[str, Any]) -> str:
    """Choose model based on current token budget position."""
    return FALLBACK_TEXT_MODEL if near_token_budget(cost_controls) else PRIMARY_TEXT_MODEL


def apply_text_tokens(cost_controls: Mapping[str, Any], response: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Increment token counters from a text tool response and recompute budget flag."""
    normalized = normalize_cost_controls(cost_controls)
    delta = extract_total_tokens_from_text_response(response)
    normalized["tokens_used_this_session"] = int(normalized["tokens_used_this_session"]) + delta
    if token_budget_exceeded(normalized):
        normalized["budget_exceeded"] = True
    return normalized


def search_cap_reached(cost_controls: Mapping[str, Any]) -> bool:
    normalized = normalize_cost_controls(cost_controls)
    return int(normalized["search_queries_used_this_session"]) >= int(
        normalized["search_query_cap_per_session"]
    )


def image_cap_reached(cost_controls: Mapping[str, Any]) -> bool:
    normalized = normalize_cost_controls(cost_controls)
    return int(normalized["image_generations_used_this_session"]) >= int(
        normalized["image_generation_cap_per_session"]
    )


def retry_cap_reached(cost_controls: Mapping[str, Any]) -> bool:
    normalized = normalize_cost_controls(cost_controls)
    return int(normalized["total_retries_used_this_session"]) >= int(
        normalized["max_total_retries_per_session"]
    )


__all__ = [
    "DEFAULT_TOKEN_BUDGET_PER_SESSION",
    "DEFAULT_SEARCH_QUERY_CAP_PER_SESSION",
    "DEFAULT_IMAGE_GENERATION_CAP_PER_SESSION",
    "DEFAULT_MAX_TOTAL_RETRIES_PER_SESSION",
    "PRIMARY_TEXT_MODEL",
    "FALLBACK_TEXT_MODEL",
    "normalize_cost_controls",
    "extract_total_tokens_from_text_response",
    "token_budget_exceeded",
    "near_token_budget",
    "preferred_text_model",
    "apply_text_tokens",
    "search_cap_reached",
    "image_cap_reached",
    "retry_cap_reached",
]
