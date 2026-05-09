"""Compatibility adapter for agent-facing text generation calls."""

from __future__ import annotations

from typing import Any, Dict, Optional

from contentblitz.config import RETRY_POLICY
from contentblitz.tools.generate_text import generate_text as _core_generate_text


def generate_text(
    prompt: str,
    agent_key: str,
    model: str = "gpt-4o",
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
    attempt_limit = int(RETRY_POLICY.get(agent_key, 0)) + 1 if agent_key in RETRY_POLICY else 0
    result = _core_generate_text(
        prompt=prompt,
        agent_key=agent_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "prompt": prompt,
        "agent_key": agent_key,
        "model_requested": model,
        "fallback_model": "gpt-4o-mini",
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
        },
    }
