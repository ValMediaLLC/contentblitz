"""Centralized text-model policy resolution for ContentBlitz agents."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Mapping

from contentblitz.config import MODEL_FALLBACKS

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:-]{2,80}$")
_ENV_POLICY_PREFIX = "CONTENTBLITZ_TEXT_MODEL_"
_GLOBAL_PROVIDER_ENV = "CONTENTBLITZ_TEXT_PROVIDER"
_GLOBAL_DEFAULT_ENV = "CONTENTBLITZ_TEXT_MODEL_DEFAULT"
_GLOBAL_FALLBACK_ENV = "CONTENTBLITZ_TEXT_MODEL_FALLBACK"
_GLOBAL_DEFAULT_ENV_ALIAS = "CONTENTBLITZ_DEFAULT_TEXT_MODEL"
_AGENT_POLICY_ENV = "CONTENTBLITZ_AGENT_MODEL_POLICY"
_SUPPORTED_TEXT_PROVIDERS = {"openai", "anthropic"}
_DEFAULT_TEXT_PROVIDER = "openai"

_OPENAI_DEFAULT_MODEL = str(
    MODEL_FALLBACKS.get("primary_text_model", "gpt-4o")
).strip()
_OPENAI_FALLBACK_MODEL = str(
    MODEL_FALLBACKS.get("fallback_text_model", "gpt-4o-mini")
).strip()
_ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-latest"
_ANTHROPIC_FALLBACK_MODEL = "claude-3-5-haiku-latest"

_GLOBAL_DEFAULT_MODEL = _OPENAI_DEFAULT_MODEL
_GLOBAL_FALLBACK_MODEL = str(
    MODEL_FALLBACKS.get("fallback_text_model", "gpt-4o-mini")
).strip()
_HIGH_QUALITY_DEFAULT = _GLOBAL_DEFAULT_MODEL or "gpt-4o"
_LOW_COST_DEFAULT = _GLOBAL_FALLBACK_MODEL or "gpt-4o-mini"


@dataclass(frozen=True)
class TextModelPolicyEntry:
    default_provider: str
    default_model: str
    fallback_provider: str
    fallback_model: str


@dataclass(frozen=True)
class ProviderModelSelection:
    provider: str
    model: str
    fallback_provider: str
    fallback_model: str


_DEFAULT_TEXT_MODEL_POLICY: Dict[str, TextModelPolicyEntry] = {
    "default": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_HIGH_QUALITY_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Fast, low-cost classifier.
    "query_handler": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_LOW_COST_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Planning/synthesis still benefits from higher default quality.
    "research_agent": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_HIGH_QUALITY_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "content_strategist": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_LOW_COST_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "blog_writer": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_LOW_COST_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "linkedin_writer": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_LOW_COST_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    "image_agent": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_LOW_COST_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Deterministic validator can opt into LLM checks when needed.
    "quality_validator": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_LOW_COST_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Optional rewrite path can be configured to higher quality as needed.
    "retry_rewrite": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_HIGH_QUALITY_DEFAULT,
        fallback_provider="openai",
        fallback_model=_LOW_COST_DEFAULT,
    ),
    # Clarification flow currently uses query-handler intent style.
    "clarification": TextModelPolicyEntry(
        default_provider="openai",
        default_model=_LOW_COST_DEFAULT,
        fallback_provider="openai",
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


def _safe_provider_name(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip().lower()
    if not candidate:
        return None
    if candidate not in _SUPPORTED_TEXT_PROVIDERS:
        return None
    return candidate


def _agent_env_key(agent_key: str, suffix: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", _normalize_agent_key(agent_key)).strip("_")
    if not safe:
        safe = "default"
    return f"{_ENV_POLICY_PREFIX}{safe.upper()}_{suffix}"


def _read_first_valid_model_env(*names: str) -> str | None:
    for name in names:
        model_name = _safe_model_name(os.getenv(name))
        if model_name:
            return model_name
    return None


def _provider_default_model(provider: str) -> str:
    if provider == "anthropic":
        return _ANTHROPIC_DEFAULT_MODEL
    return _safe_model_name(_OPENAI_DEFAULT_MODEL) or "gpt-4o"


def _provider_fallback_model(provider: str) -> str:
    if provider == "anthropic":
        return _ANTHROPIC_FALLBACK_MODEL
    return _safe_model_name(_OPENAI_FALLBACK_MODEL) or "gpt-4o-mini"


def _default_fallback_provider(provider: str) -> str:
    # Global/provider-level defaults should remain provider-consistent unless
    # an explicit fallback_provider override is supplied.
    return provider


def _parse_agent_policy_env() -> Dict[str, Dict[str, str]]:
    raw = os.getenv(_AGENT_POLICY_ENV)
    if raw is None:
        return {}
    payload = str(raw).strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, Mapping):
        return {}
    normalized: Dict[str, Dict[str, str]] = {}
    for key, value in parsed.items():
        agent_key = _normalize_agent_key(str(key))
        if not isinstance(value, Mapping):
            continue
        normalized_entry: Dict[str, str] = {}
        for policy_key in (
            "provider",
            "model",
            "fallback_provider",
            "fallback_model",
        ):
            policy_value = value.get(policy_key)
            if policy_value is None:
                continue
            normalized_entry[policy_key] = str(policy_value).strip()
        if normalized_entry:
            normalized[agent_key] = normalized_entry
    return normalized


def build_text_model_policy() -> Dict[str, TextModelPolicyEntry]:
    """Build the effective text-model policy from defaults and env overrides."""
    explicit_global_provider = _safe_provider_name(os.getenv(_GLOBAL_PROVIDER_ENV))
    base_provider = explicit_global_provider or _DEFAULT_TEXT_PROVIDER
    explicit_global_default_model = _read_first_valid_model_env(
        _GLOBAL_DEFAULT_ENV_ALIAS,
        _GLOBAL_DEFAULT_ENV,
    )
    explicit_global_fallback_model = _safe_model_name(os.getenv(_GLOBAL_FALLBACK_ENV))
    policy_overrides = _parse_agent_policy_env()

    resolved: Dict[str, TextModelPolicyEntry] = {}
    for agent_key, entry in _DEFAULT_TEXT_MODEL_POLICY.items():
        agent_override = policy_overrides.get(_normalize_agent_key(agent_key), {})
        env_default = _safe_model_name(os.getenv(_agent_env_key(agent_key, "DEFAULT")))
        env_fallback = _safe_model_name(
            os.getenv(_agent_env_key(agent_key, "FALLBACK"))
        )

        provider = (
            _safe_provider_name(agent_override.get("provider"))
            or base_provider
            or _safe_provider_name(entry.default_provider)
            or _DEFAULT_TEXT_PROVIDER
        )
        fallback_provider = (
            _safe_provider_name(agent_override.get("fallback_provider"))
            or _default_fallback_provider(provider)
            or _safe_provider_name(entry.fallback_provider)
            or _DEFAULT_TEXT_PROVIDER
        )

        default_model = (
            env_default
            or _safe_model_name(agent_override.get("model"))
            or explicit_global_default_model
            or (
                _safe_model_name(entry.default_model)
                if provider == _safe_provider_name(entry.default_provider)
                else None
            )
            or _provider_default_model(provider)
            or _GLOBAL_DEFAULT_MODEL
        )
        fallback_model = (
            env_fallback
            or _safe_model_name(agent_override.get("fallback_model"))
            or explicit_global_fallback_model
            or (
                _safe_model_name(entry.fallback_model)
                if fallback_provider == _safe_provider_name(entry.fallback_provider)
                else None
            )
            or _provider_fallback_model(fallback_provider)
            or _GLOBAL_FALLBACK_MODEL
            or default_model
        )

        resolved[agent_key] = TextModelPolicyEntry(
            default_provider=provider,
            default_model=default_model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )

    # Safe fallback for unknown callers.
    unknown_provider = base_provider or _DEFAULT_TEXT_PROVIDER
    unknown_fallback_provider = _default_fallback_provider(unknown_provider)
    resolved["unknown"] = TextModelPolicyEntry(
        default_provider=unknown_provider,
        default_model=(
            explicit_global_default_model or _provider_default_model(unknown_provider)
        ),
        fallback_provider=unknown_fallback_provider,
        fallback_model=(
            explicit_global_fallback_model
            or _provider_fallback_model(unknown_fallback_provider)
            or explicit_global_default_model
            or _provider_default_model(unknown_provider)
        ),
    )
    return resolved


def resolve_text_provider_model(
    agent_key: str | None,
    *,
    near_budget: bool,
    policy: Mapping[str, TextModelPolicyEntry] | None = None,
) -> ProviderModelSelection:
    """Resolve provider+model pair for an agent key with budget-aware behavior."""
    effective_policy = policy or build_text_model_policy()
    normalized = _normalize_agent_key(agent_key)
    entry = (
        effective_policy.get(normalized)
        or effective_policy.get("default")
        or effective_policy.get("unknown")
    )
    if entry is None:
        provider = _DEFAULT_TEXT_PROVIDER
        model = _provider_fallback_model(provider)
        return ProviderModelSelection(
            provider=provider,
            model=model,
            fallback_provider=provider,
            fallback_model=model,
        )

    default_provider = (
        _safe_provider_name(entry.default_provider) or _DEFAULT_TEXT_PROVIDER
    )
    fallback_provider = (
        _safe_provider_name(entry.fallback_provider)
        or _default_fallback_provider(default_provider)
        or _DEFAULT_TEXT_PROVIDER
    )
    default_model = _safe_model_name(entry.default_model) or _provider_default_model(
        default_provider
    )
    fallback_model = (
        _safe_model_name(entry.fallback_model)
        or _provider_fallback_model(fallback_provider)
        or default_model
    )

    if near_budget:
        selected_provider = fallback_provider
        selected_model = fallback_model
    else:
        selected_provider = default_provider
        selected_model = default_model

    return ProviderModelSelection(
        provider=selected_provider,
        model=selected_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
    )


def resolve_text_model(
    agent_key: str | None,
    *,
    near_budget: bool,
    policy: Mapping[str, TextModelPolicyEntry] | None = None,
) -> str:
    """Resolve the model for an agent key, with budget-aware fallback selection."""
    selection = resolve_text_provider_model(
        agent_key,
        near_budget=near_budget,
        policy=policy,
    )
    return selection.model


__all__ = [
    "ProviderModelSelection",
    "TextModelPolicyEntry",
    "KNOWN_TEXT_MODEL_AGENT_KEYS",
    "build_text_model_policy",
    "resolve_text_provider_model",
    "resolve_text_model",
]
