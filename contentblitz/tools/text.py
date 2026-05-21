"""Compatibility adapter for agent-facing text generation calls."""

from __future__ import annotations

from typing import Any, Dict, Optional

from contentblitz.config import RETRY_POLICY
from contentblitz.core.model_policy import resolve_text_provider_model
from contentblitz.tools.generate_text import generate_text as _core_generate_text

_RETRY_POLICY_AGENT_ALIASES = {
    "clarification": "query_handler",
}


def _retry_policy_agent_key(agent_key: str) -> str:
    normalized = str(agent_key).strip()
    if not normalized:
        return normalized
    return _RETRY_POLICY_AGENT_ALIASES.get(normalized, normalized)


def generate_text(
    prompt: str,
    agent_key: str,
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Legacy dict contract used by existing agents.

    Internally this delegates to the typed OpenAI-backed tool in
    `contentblitz.tools.generate_text`.
    """
    selection = resolve_text_provider_model(
        agent_key,
        near_budget=False,
    )
    requested_model = str(model).strip() if model is not None else ""
    if not requested_model:
        requested_model = selection.model
    fallback_model = selection.fallback_model or requested_model

    retry_policy_agent = _retry_policy_agent_key(agent_key)
    attempt_limit = (
        int(RETRY_POLICY.get(retry_policy_agent, 0)) + 1
        if retry_policy_agent in RETRY_POLICY
        else 0
    )
    result = _core_generate_text(
        prompt=prompt,
        agent_key=agent_key,
        model=requested_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "prompt": prompt,
        "agent_key": agent_key,
        "provider_requested": selection.provider,
        "model_requested": requested_model,
        "fallback_provider": selection.fallback_provider,
        "fallback_model": fallback_model,
        "attempt_limit": attempt_limit,
        "metadata": metadata or {},
        "output": result.text,
        "used_external_api": not result.degraded,
        "model": result.model,
        "provider": result.provider,
        "degraded": result.degraded,
        "error": result.error,
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "cache_creation_input_tokens": result.cache_creation_input_tokens,
            "cache_read_input_tokens": result.cache_read_input_tokens,
        },
    }
