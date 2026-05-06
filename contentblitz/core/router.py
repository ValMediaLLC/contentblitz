"""Core retry routing utilities for deterministic workflow execution."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping

from contentblitz.config import RETRY_POLICY


def increment_retry_count(state: MutableMapping[str, Any], agent_key: str) -> int:
    """Increment retry count for an agent and return the updated count."""
    retry_counts = state.setdefault("retry_counts", {})
    retry_counts[agent_key] = int(retry_counts.get(agent_key, 0)) + 1
    return retry_counts[agent_key]


def retry_remaining(state: MutableMapping[str, Any], agent_key: str) -> bool:
    """True when agent still has retries available under configured policy."""
    used = int(state.get("retry_counts", {}).get(agent_key, 0))
    limit = int(RETRY_POLICY.get(agent_key, 0))
    return used <= limit


def route_with_retry(
    state: MutableMapping[str, Any],
    agent_key: str,
    retry_node: str,
    exhausted_node: str = "error_handler_node",
) -> str:
    """Increment counter before deciding route, per spec routing rules."""
    increment_retry_count(state, agent_key)
    return retry_node if retry_remaining(state, agent_key) else exhausted_node


def retry_snapshot(state: MutableMapping[str, Any]) -> Dict[str, int]:
    """Return a copy of retry counters."""
    return dict(state.get("retry_counts", {}))

