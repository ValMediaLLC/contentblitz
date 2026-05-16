"""Cache backend adapters used by the cache contract."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


class InMemoryCacheBackend:
    """Shared in-process cache backend with TTL support."""

    def __init__(self, *, now_fn: Callable[[], int] | None = None) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._now_fn = now_fn or (lambda: int(time.time()))

    def _now(self) -> int:
        return int(self._now_fn())

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not isinstance(entry, Mapping):
            return None
        expires_at = entry.get("expires_at")
        if isinstance(expires_at, (int, float)) and int(expires_at) <= self._now():
            self._store.pop(key, None)
            return None
        if "value" not in entry:
            return None
        return deepcopy(entry.get("value"))

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return False

        safe_ttl = max(0, int(ttl_seconds))
        now_epoch = self._now()
        expires_at: int | None = None if safe_ttl == 0 else now_epoch + safe_ttl
        self._store[key] = {
            "key": key,
            "value": deepcopy(value),
            "created_at": now_epoch,
            "expires_at": expires_at,
            "metadata": dict(metadata or {}),
        }
        return True

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def clear(self) -> bool:
        self._store.clear()
        return True


class SQLiteCacheBackend:
    """Local SQLite cache backend with TTL support."""

    def __init__(self, path: Path, *, now_fn: Callable[[], int] | None = None) -> None:
        self._path = Path(path)
        self._now_fn = now_fn or (lambda: int(time.time()))
        self._ready = False

    def _now(self) -> int:
        return int(self._now_fn())

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self._path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        expires_at INTEGER,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            self._ready = True
            return True
        except (OSError, sqlite3.Error):
            return False

    def get(self, key: str) -> Optional[Any]:
        if not self._ensure_ready():
            return None
        try:
            with closing(sqlite3.connect(self._path)) as conn:
                row = conn.execute(
                    "SELECT value_json, expires_at FROM cache_entries WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                value_json, expires_at = row
                if (
                    isinstance(expires_at, (int, float))
                    and int(expires_at) <= self._now()
                ):
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return None
                return json.loads(value_json)
        except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
            return None

    def set(
        self,
        key: str,
        value: Any,
        *,
        ttl_seconds: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        if not self._ensure_ready():
            return False
        try:
            value_json = json.dumps(value)
            metadata_json = json.dumps(dict(metadata or {}))
        except (TypeError, ValueError):
            return False

        safe_ttl = max(0, int(ttl_seconds))
        now_epoch = self._now()
        expires_at: int | None = None if safe_ttl == 0 else now_epoch + safe_ttl
        try:
            with closing(sqlite3.connect(self._path)) as conn:
                conn.execute(
                    """
                    INSERT INTO cache_entries (key, value_json, created_at, expires_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json=excluded.value_json,
                        created_at=excluded.created_at,
                        expires_at=excluded.expires_at,
                        metadata_json=excluded.metadata_json
                    """,
                    (key, value_json, now_epoch, expires_at, metadata_json),
                )
                conn.commit()
            return True
        except sqlite3.Error:
            return False

    def delete(self, key: str) -> bool:
        if not self._ensure_ready():
            return False
        try:
            with closing(sqlite3.connect(self._path)) as conn:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
            return bool(cursor.rowcount)
        except sqlite3.Error:
            return False

    def clear(self) -> bool:
        if not self._ensure_ready():
            return False
        try:
            with closing(sqlite3.connect(self._path)) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
            return True
        except sqlite3.Error:
            return False
