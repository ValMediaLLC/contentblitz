"""In-memory cache helpers for research workflow scaffolding."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


def normalize_query(query: str) -> str:
    """Normalize query text before hashing."""
    return " ".join(str(query).strip().lower().split())


def build_research_cache_key(query: str, depth: str = "standard") -> str:
    """
    Build a cache key that follows spec:
    research:{sha256(normalized_query)}:{depth}
    """
    normalized = normalize_query(query)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"research:{digest}:{depth}"


def get_cached_research(
    state: Mapping[str, Any],
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    """Read a cached research payload if cache is enabled and key exists."""
    cache_meta = state.get("cache_metadata", {})
    if not isinstance(cache_meta, Mapping) or not bool(cache_meta.get("enabled", False)):
        return None

    tool_outputs = state.get("tool_outputs", {})
    if not isinstance(tool_outputs, Mapping):
        return None

    cache_store = tool_outputs.get("research_cache", {})
    if not isinstance(cache_store, Mapping):
        return None

    payload = cache_store.get(cache_key)
    if not isinstance(payload, dict):
        return None
    return deepcopy(payload)


def set_cached_research(
    state: Mapping[str, Any],
    cache_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Return state updates that persist a research payload in cache."""
    tool_outputs = deepcopy(state.get("tool_outputs", {}))
    if not isinstance(tool_outputs, dict):
        tool_outputs = {}

    cache_store = deepcopy(tool_outputs.get("research_cache", {}))
    if not isinstance(cache_store, dict):
        cache_store = {}
    cache_store[cache_key] = deepcopy(payload)
    tool_outputs["research_cache"] = cache_store

    cache_meta = deepcopy(state.get("cache_metadata", {}))
    if not isinstance(cache_meta, dict):
        cache_meta = {}
    keys = list(cache_meta.get("keys", [])) if isinstance(cache_meta.get("keys", []), list) else []
    if cache_key not in keys:
        keys.append(cache_key)
    cache_meta["keys"] = keys

    return {
        "tool_outputs": tool_outputs,
        "cache_metadata": cache_meta,
    }

