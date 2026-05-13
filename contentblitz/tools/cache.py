"""Cache helpers for production-capable research caching."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from contentblitz.config import CACHE_METADATA_DEFAULTS
from contentblitz.core.cache_keys import (
    build_research_cache_key as _build_research_cache_key,
    normalize_query as _normalize_query,
)
from contentblitz.tools.cache_backends import InMemoryCacheBackend, SQLiteCacheBackend

_DEFAULT_SQLITE_PATH = ".tmp/contentblitz_cache.sqlite3"
_IN_MEMORY_BACKEND = InMemoryCacheBackend(now_fn=lambda: _now_epoch_seconds())
_SQLITE_BACKENDS: Dict[str, SQLiteCacheBackend] = {}


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
    default_ttl = _default_ttl_seconds()
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


def _default_ttl_seconds() -> int:
    default_ttl = int(CACHE_METADATA_DEFAULTS.get("ttl_seconds", 1800))
    raw = os.getenv("CONTENTBLITZ_CACHE_TTL_SECONDS")
    if raw is None:
        return default_ttl
    try:
        return max(0, int(raw.strip()))
    except (TypeError, ValueError):
        return default_ttl


def _configured_backend_name() -> str:
    raw = str(os.getenv("CONTENTBLITZ_CACHE_BACKEND", "in_memory")).strip().lower()
    if raw in {"in_memory", "in-memory", "memory", "inmemory"}:
        return "in_memory"
    if raw == "sqlite":
        return "sqlite"
    return "in_memory"


def _resolve_sqlite_path() -> Path:
    raw = str(os.getenv("CONTENTBLITZ_CACHE_SQLITE_PATH", _DEFAULT_SQLITE_PATH)).strip()
    candidate = Path(raw) if raw else Path(_DEFAULT_SQLITE_PATH)
    root = Path.cwd().resolve()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("SQLite cache path must stay within project root.") from exc
    return resolved


def _active_backend() -> tuple[str, InMemoryCacheBackend | SQLiteCacheBackend]:
    backend_name = _configured_backend_name()
    if backend_name == "sqlite":
        try:
            sqlite_path = _resolve_sqlite_path()
        except (ValueError, OSError):
            return "in_memory", _IN_MEMORY_BACKEND
        cache_key = str(sqlite_path)
        backend = _SQLITE_BACKENDS.get(cache_key)
        if backend is None:
            backend = SQLiteCacheBackend(sqlite_path, now_fn=lambda: _now_epoch_seconds())
            _SQLITE_BACKENDS[cache_key] = backend
        return "sqlite", backend
    return "in_memory", _IN_MEMORY_BACKEND


def get_cache_backend_name() -> str:
    """Return the effective cache backend name after safe fallback handling."""
    name, _ = _active_backend()
    return name


def set_cache(key: str, value: Any, ttl_seconds: int = 1800) -> bool:
    """Persist a value in the active cache backend."""
    safe_key = _safe_key(key)
    if not safe_key:
        return False
    if not _is_json_serializable(value):
        return False

    try:
        safe_ttl = int(ttl_seconds)
    except (TypeError, ValueError):
        safe_ttl = _default_ttl_seconds()
    safe_ttl = max(0, safe_ttl)
    backend_name, backend = _active_backend()
    return backend.set(
        safe_key,
        deepcopy(value),
        ttl_seconds=safe_ttl,
        metadata={"backend": backend_name},
    )


def get_cache(key: str) -> Optional[Any]:
    """Read a value from the active cache backend if present and not expired."""
    safe_key = _safe_key(key)
    if not safe_key:
        return None
    _, backend = _active_backend()
    return backend.get(safe_key)


def delete_cache(key: str) -> bool:
    """Delete a cache key from the active backend."""
    safe_key = _safe_key(key)
    if not safe_key:
        return False
    _, backend = _active_backend()
    return backend.delete(safe_key)


def clear_cache() -> None:
    """Clear all known cache backend entries."""
    _IN_MEMORY_BACKEND.clear()
    for backend in _SQLITE_BACKENDS.values():
        backend.clear()


def _cache_metadata_update(state: Mapping[str, Any], cache_key: str) -> Dict[str, Any]:
    cache_meta = deepcopy(_safe_dict(state.get("cache_metadata", {})))
    if not isinstance(cache_meta, dict):
        cache_meta = {}

    cache_meta.setdefault("enabled", bool(CACHE_METADATA_DEFAULTS.get("enabled", True)))
    cache_meta["ttl_seconds"] = _cache_ttl_seconds(state)
    cache_meta["backend"] = get_cache_backend_name()

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
    "get_cache_backend_name",
    "get_cached_research",
    "set_cached_research",
    "touch_cached_research_key",
]
