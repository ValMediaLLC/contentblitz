"""Centralized text-model policy resolution for ContentBlitz agents."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Mapping

from contentblitz.config import MODEL_FALLBACKS

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:-]{2,80}$")
_ENV_POLICY_PREFIX = "CONTENTBLITZ_TEXT_MODEL_"
_GLOBAL_DEFAULT_ENV = "CONTENTBLITZ_TEXT_MODEL_DEFAULT"
_GLOBAL_FALLBACK_ENV = "CONTENTBLITZ_TEXT_MODEL_FALLBACK"

_GLOBAL_DEFAULT_MODEL = str(MODEL_FALLBACKS.get("primary_text_model", "gpt-4o")).strip()
_GLOBAL_FALLBACK_MODEL = str(
    MODEL_FALLBACKS.get("fallback_text_model", "gpt-4o-mini")
).strip()
_HIGH_QUALITY_DEFAULT = _GLOBAL_DEFAULT_MODEL or "gpt-4o"
_LOW_COST_DEFAULT = _GLOBAL_FALLBACK_MODEL or "gpt-4o-mini"


@dataclass(frozen=True)
class TextModelPolicyEntry:
    default_model: str
    fallback_model: str


_DEFAULT_TEXT_MODEL_POLICY: Dict[str, TextModelPolicyEntry] = {
    "default": TextModelPolicyEntry(
        default_model=_HIGH_QUALITY_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Fast, low-cost classifier.
    "query_handler": TextModelPolicyEntry(
        default_model=_LOW_COST_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Planning/synthesis still benefits from higher default quality.
    "research_agent": TextModelPolicyEntry(
        default_model=_HIGH_QUALITY_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "content_strategist": TextModelPolicyEntry(
        default_model=_LOW_COST_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "blog_writer": TextModelPolicyEntry(
        default_model=_LOW_COST_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "linkedin_writer": TextModelPolicyEntry(
        default_model=_LOW_COST_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "image_agent": TextModelPolicyEntry(
        default_model=_LOW_COST_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Deterministic validator can opt into LLM checks when needed.
    "quality_validator": TextModelPolicyEntry(
        default_model=_LOW_COST_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Optional rewrite path can be configured to higher quality as needed.
    "retry_rewrite": TextModelPolicyEntry(
        default_model=_HIGH_QUALITY_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Clarification flow currently uses query-handler intent style.
    "clarification": TextModelPolicyEntry(
        default_model=_LOW_COST_DEFAULT,
        fallback_model=_LOW_COST_DEFAULT,
    ),
}

KNOWN_TEXT_MODEL_AGENT_KEYS = tuple(_DEFAULT_TEXT_MODEL_POLICY.keys())


def _safe_model_name(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    if not _MODEL_NAME_RE.fullmatch(candidate):
        return None
    return candidate


def _normalize_agent_key(agent_key: str | None) -> str:
    normalized = str(agent_key or "").strip().lower()
    return normalized or "default"


def _agent_env_key(agent_key: str, suffix: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", _normalize_agent_key(agent_key)).strip("_")
    if not safe:
        safe = "default"
    return f"{_ENV_POLICY_PREFIX}{safe.upper()}_{suffix}"


def _resolved_entry(
    *,
    agent_key: str,
    base_entry: TextModelPolicyEntry,
    explicit_global_default_model: str | None,
    explicit_global_fallback_model: str | None,
    base_global_default_model: str,
    base_global_fallback_model: str,
) -> TextModelPolicyEntry:
    env_default = _safe_model_name(os.getenv(_agent_env_key(agent_key, "DEFAULT")))
    env_fallback = _safe_model_name(os.getenv(_agent_env_key(agent_key, "FALLBACK")))
    default_model = (
        env_default
        or explicit_global_default_model
        or _safe_model_name(base_entry.default_model)
        or base_global_default_model
    )
    fallback_model = (
        env_fallback
        or explicit_global_fallback_model
        or _safe_model_name(base_entry.fallback_model)
        or base_global_fallback_model
        or default_model
    )
    return TextModelPolicyEntry(
        default_model=default_model,
        fallback_model=fallback_model,
    )


def build_text_model_policy() -> Dict[str, TextModelPolicyEntry]:
    """Build the effective text-model policy from defaults and env overrides."""
    explicit_global_default_model = _safe_model_name(os.getenv(_GLOBAL_DEFAULT_ENV))
    explicit_global_fallback_model = _safe_model_name(os.getenv(_GLOBAL_FALLBACK_ENV))
    base_global_default_model = _safe_model_name(_GLOBAL_DEFAULT_MODEL) or "gpt-4o"
    base_global_fallback_model = (
        _safe_model_name(_GLOBAL_FALLBACK_MODEL) or "gpt-4o-mini"
    )

    resolved: Dict[str, TextModelPolicyEntry] = {}
    for agent_key, entry in _DEFAULT_TEXT_MODEL_POLICY.items():
        resolved[agent_key] = _resolved_entry(
            agent_key=agent_key,
            base_entry=entry,
            explicit_global_default_model=explicit_global_default_model,
            explicit_global_fallback_model=explicit_global_fallback_model,
            base_global_default_model=base_global_default_model,
            base_global_fallback_model=base_global_fallback_model,
        )

    # Safe fallback for unknown callers.
    resolved["unknown"] = TextModelPolicyEntry(
        default_model=explicit_global_default_model or base_global_default_model,
        fallback_model=(
            explicit_global_fallback_model
            or base_global_fallback_model
            or explicit_global_default_model
            or base_global_default_model
        ),
    )
    return resolved


def resolve_text_model(
    agent_key: str | None,
    *,
    near_budget: bool,
    policy: Mapping[str, TextModelPolicyEntry] | None = None,
) -> str:
    """Resolve the model for an agent key, with budget-aware fallback selection."""
    effective_policy = policy or build_text_model_policy()
    normalized = _normalize_agent_key(agent_key)
    entry = (
        effective_policy.get(normalized)
        or effective_policy.get("default")
        or effective_policy.get("unknown")
    )
    if entry is None:
        return _GLOBAL_FALLBACK_MODEL
    if near_budget:
        return entry.fallback_model or entry.default_model
    return entry.default_model or entry.fallback_model


__all__ = [
    "TextModelPolicyEntry",
    "KNOWN_TEXT_MODEL_AGENT_KEYS",
    "build_text_model_policy",
    "resolve_text_model",
]
