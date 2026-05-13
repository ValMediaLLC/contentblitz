from __future__ import annotations

from pathlib import Path

import pytest

from contentblitz.tools import cache as cache_module


@pytest.fixture(autouse=True)
def _reset_cache_and_env(monkeypatch) -> None:
    monkeypatch.delenv("CONTENTBLITZ_CACHE_BACKEND", raising=False)
    monkeypatch.delenv("CONTENTBLITZ_CACHE_SQLITE_PATH", raising=False)
    monkeypatch.delenv("CONTENTBLITZ_CACHE_TTL_SECONDS", raising=False)
    cache_module.clear_cache()
    yield
    cache_module.clear_cache()


def test_sqlite_backend_set_get_delete_clear(tmp_path, monkeypatch) -> None:
    sqlite_path = f".tmp/{tmp_path.name}_cache.sqlite3"
    monkeypatch.setenv("CONTENTBLITZ_CACHE_BACKEND", "sqlite")
    monkeypatch.setenv("CONTENTBLITZ_CACHE_SQLITE_PATH", sqlite_path)

    key = cache_module.build_research_cache_key("sqlite roundtrip")
    payload = {"research_data": {"status": "complete"}, "sources": []}

    assert cache_module.get_cache_backend_name() == "sqlite"
    assert cache_module.set_cache(key, payload, ttl_seconds=30) is True
    assert cache_module.get_cache(key) == payload
    assert cache_module.delete_cache(key) is True
    assert cache_module.get_cache(key) is None

    assert cache_module.set_cache(key, payload, ttl_seconds=30) is True
    cache_module.clear_cache()
    assert cache_module.get_cache(key) is None
    assert Path(sqlite_path).exists()


def test_sqlite_ttl_expiration_is_honored(tmp_path, monkeypatch) -> None:
    sqlite_path = f".tmp/{tmp_path.name}_ttl.sqlite3"
    monkeypatch.setenv("CONTENTBLITZ_CACHE_BACKEND", "sqlite")
    monkeypatch.setenv("CONTENTBLITZ_CACHE_SQLITE_PATH", sqlite_path)

    key = cache_module.build_research_cache_key("sqlite ttl")
    payload = {"research_data": {"status": "complete"}, "sources": []}

    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 1000)
    assert cache_module.set_cache(key, payload, ttl_seconds=10) is True
    assert cache_module.get_cache(key) == payload

    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 1012)
    assert cache_module.get_cache(key) is None


def test_sqlite_ttl_zero_does_not_expire(tmp_path, monkeypatch) -> None:
    sqlite_path = f".tmp/{tmp_path.name}_ttl_zero.sqlite3"
    monkeypatch.setenv("CONTENTBLITZ_CACHE_BACKEND", "sqlite")
    monkeypatch.setenv("CONTENTBLITZ_CACHE_SQLITE_PATH", sqlite_path)

    key = cache_module.build_research_cache_key("sqlite ttl zero")
    payload = {"research_data": {"status": "complete"}, "sources": []}

    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 2000)
    assert cache_module.set_cache(key, payload, ttl_seconds=0) is True
    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 999999)
    assert cache_module.get_cache(key) == payload


def test_invalid_backend_name_falls_back_to_in_memory(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_CACHE_BACKEND", "redis")
    assert cache_module.get_cache_backend_name() == "in_memory"

    key = cache_module.build_research_cache_key("fallback backend")
    payload = {"research_data": {"status": "complete"}, "sources": []}
    assert cache_module.set_cache(key, payload, ttl_seconds=30) is True
    assert cache_module.get_cache(key) == payload


def test_invalid_sqlite_path_falls_back_to_in_memory(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_CACHE_BACKEND", "sqlite")
    monkeypatch.setenv("CONTENTBLITZ_CACHE_SQLITE_PATH", "../outside_project/cache.sqlite3")
    assert cache_module.get_cache_backend_name() == "in_memory"

    key = cache_module.build_research_cache_key("invalid sqlite path")
    payload = {"research_data": {"status": "complete"}, "sources": []}
    assert cache_module.set_cache(key, payload, ttl_seconds=30) is True
    assert cache_module.get_cache(key) == payload


def test_sqlite_rejects_non_serializable_values(tmp_path, monkeypatch) -> None:
    sqlite_path = f".tmp/{tmp_path.name}_badvalue.sqlite3"
    monkeypatch.setenv("CONTENTBLITZ_CACHE_BACKEND", "sqlite")
    monkeypatch.setenv("CONTENTBLITZ_CACHE_SQLITE_PATH", sqlite_path)

    key = cache_module.build_research_cache_key("sqlite bad value")
    payload = {"research_data": {"status": "complete"}, "raw": {1, 2, 3}}
    assert cache_module.set_cache(key, payload, ttl_seconds=30) is False
    assert cache_module.get_cache(key) is None
