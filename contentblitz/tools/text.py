"""Text generation tool interface scaffold."""

from __future__ import annotations

from typing import Any, Dict, Optional

from contentblitz.config import RETRY_POLICY


def generate_text(
    prompt: str,
    agent_key: str,
    model: str = "gpt-4o",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return scaffolded output metadata without calling external APIs."""
    retries = RETRY_POLICY[agent_key] + 1
    return {
        "prompt": prompt,
        "agent_key": agent_key,
        "model_requested": model,
        "fallback_model": "gpt-4o-mini",
        "attempt_limit": retries,
        "metadata": metadata or {},
        "output": "",
        "used_external_api": False,
    }

