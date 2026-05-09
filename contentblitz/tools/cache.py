"""In-memory cache helpers for production-capable research caching."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from contentblitz.config import CACHE_METADATA_DEFAULTS
from contentblitz.core.cache_keys import (
    build_research_cache_key as _build_research_cache_key,
    normalize_query as _normalize_query,
)

# Shared process-level cache backend used across separate state objects/sessions.
_CACHE_STORE: Dict[str, Dict[str, Any]] = {}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_key(key: Any) -> str:
    return str(key or "").strip()


def build_research_cache_key(query: str, depth: str = "standard") -> str:
    """Compatibility wrapper around canonical cache key builder."""
    return _build_research_cache_key(query=query, depth=depth)


def normalize_query(query: str) -> str:
    """Compatibility wrapper around canonical query normalizer."""
    return _normalize_query(query=query)


def _cache_enabled(state: Mapping[str, Any]) -> bool:
    cache_meta = state.get("cache_metadata", {})
    if not isinstance(cache_meta, Mapping):
        return bool(CACHE_METADATA_DEFAULTS.get("enabled", True))
    return bool(cache_meta.get("enabled", CACHE_METADATA_DEFAULTS.get("enabled", True)))


def _cache_ttl_seconds(state: Mapping[str, Any]) -> int:
    cache_meta = state.get("cache_metadata", {})
    default_ttl = int(CACHE_METADATA_DEFAULTS.get("ttl_seconds", 1800))
    if not isinstance(cache_meta, Mapping):
        return default_ttl
    raw = cache_meta.get("ttl_seconds", default_ttl)
    if isinstance(raw, bool):
        return default_ttl
    if isinstance(raw, int):
        return max(0, raw)
    if isinstance(raw, float):
        return max(0, int(raw))
    return default_ttl


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _now_epoch_seconds() -> int:
    return int(time.time())


def set_cache(key: str, value: Any, ttl_seconds: int = 1800) -> bool:
    """Persist a value in the shared process-level cache backend."""
    safe_key = _safe_key(key)
    if not safe_key:
        return False
    if not _is_json_serializable(value):
        return False

    try:
        safe_ttl = int(ttl_seconds)
    except (TypeError, ValueError):
        safe_ttl = int(CACHE_METADATA_DEFAULTS.get("ttl_seconds", 1800))
    safe_ttl = max(0, safe_ttl)
    now_epoch = _now_epoch_seconds()
    expires_at: Optional[int] = None if safe_ttl == 0 else now_epoch + safe_ttl
    _CACHE_STORE[safe_key] = {
        "value": deepcopy(value),
        "cached_at": now_epoch,
        "expires_at": expires_at,
    }
    return True


def get_cache(key: str) -> Optional[Any]:
    """Read a value from the shared cache backend if present and not expired."""
    safe_key = _safe_key(key)
    if not safe_key:
        return None
    entry = _CACHE_STORE.get(safe_key)
    if not isinstance(entry, Mapping):
        return None

    expires_at = entry.get("expires_at")
    if isinstance(expires_at, (int, float)) and int(expires_at) <= _now_epoch_seconds():
        _CACHE_STORE.pop(safe_key, None)
        return None

    if "value" not in entry:
        return None
    return deepcopy(entry.get("value"))


def delete_cache(key: str) -> bool:
    """Delete a cache key from the shared backend."""
    safe_key = _safe_key(key)
    if not safe_key:
        return False
    return _CACHE_STORE.pop(safe_key, None) is not None


def clear_cache() -> None:
    """Clear all in-memory cache entries."""
    _CACHE_STORE.clear()


def _cache_metadata_update(state: Mapping[str, Any], cache_key: str) -> Dict[str, Any]:
    cache_meta = deepcopy(_safe_dict(state.get("cache_metadata", {})))
    if not isinstance(cache_meta, dict):
        cache_meta = {}

    cache_meta.setdefault("enabled", bool(CACHE_METADATA_DEFAULTS.get("enabled", True)))
    cache_meta.setdefault("ttl_seconds", int(CACHE_METADATA_DEFAULTS.get("ttl_seconds", 1800)))
    cache_meta.setdefault("backend", str(CACHE_METADATA_DEFAULTS.get("backend", "in_memory")))

    keys = list(cache_meta.get("keys", [])) if isinstance(cache_meta.get("keys", []), list) else []
    if cache_key not in keys:
        keys.append(cache_key)
    cache_meta["keys"] = keys
    return {"cache_metadata": cache_meta}


def touch_cached_research_key(state: Mapping[str, Any], cache_key: str) -> Dict[str, Any]:
    """Return state updates that record cache key visibility on cache hits."""
    if not _cache_enabled(state):
        return {}
    return _cache_metadata_update(state, cache_key)


def _get_legacy_state_cached_research(
    state: Mapping[str, Any],
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Backward-compatible read path for historical state-embedded cache payloads.

    This supports existing test fixtures while the shared process cache is primary.
    """
    tool_outputs = state.get("tool_outputs", {})
    if not isinstance(tool_outputs, Mapping):
        return None

    cache_store = tool_outputs.get("research_cache", {})
    if not isinstance(cache_store, Mapping):
        return None

    entry = cache_store.get(cache_key)
    if not isinstance(entry, Mapping):
        if isinstance(entry, dict):
            return deepcopy(entry)
        return None

    if "value" not in entry:
        if isinstance(entry, dict):
            return deepcopy(dict(entry))
        return None

    expires_at = entry.get("expires_at")
    if isinstance(expires_at, (int, float)) and int(expires_at) <= _now_epoch_seconds():
        return None

    payload = entry.get("value")
    if not isinstance(payload, dict):
        return None
    return deepcopy(payload)


def get_cached_research(
    state: Mapping[str, Any],
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    """Read a cached research payload if cache is enabled and key exists."""
    if not _cache_enabled(state):
        return None

    payload = get_cache(cache_key)
    if isinstance(payload, dict):
        return payload

    return _get_legacy_state_cached_research(state, cache_key)


def set_cached_research(
    state: Mapping[str, Any],
    cache_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a research payload in shared cache and return metadata updates."""
    if not _cache_enabled(state):
        return {}
    if not isinstance(payload, dict):
        return {}
    if not _is_json_serializable(payload):
        return {}

    if not set_cache(cache_key, payload, ttl_seconds=_cache_ttl_seconds(state)):
        return {}

    return _cache_metadata_update(state, cache_key)


__all__ = [
    "build_research_cache_key",
    "normalize_query",
    "get_cache",
    "set_cache",
    "delete_cache",
    "clear_cache",
    "get_cached_research",
    "set_cached_research",
    "touch_cached_research_key",
]
