from __future__ import annotations

import pytest

from contentblitz.state import create_initial_state
from contentblitz.tools import cache as cache_module


@pytest.fixture(autouse=True)
def _clear_process_cache() -> None:
    cache_module.clear_cache()
    yield
    cache_module.clear_cache()


def test_normalize_query_is_deterministic() -> None:
    assert cache_module.normalize_query("  AI   Trends  2026 ") == "ai trends 2026"


def test_default_backend_remains_in_memory() -> None:
    assert cache_module.get_cache_backend_name() == "in_memory"


def test_build_research_cache_key_follows_spec_shape() -> None:
    key = cache_module.build_research_cache_key("AI market outlook", depth="deep")
    assert key.startswith("research:")
    assert key.endswith(":deep")
    assert "AI market outlook" not in key


def test_set_cache_then_get_cache_round_trip_and_copy_safety() -> None:
    key = cache_module.build_research_cache_key("cloud security")
    payload = {
        "research_data": {"status": "complete", "degraded": False},
        "sources": [{"title": "A", "url": "https://example.com"}],
    }

    assert cache_module.set_cache(key, payload, ttl_seconds=1800) is True
    cached = cache_module.get_cache(key)
    assert cached == payload

    cached["research_data"]["status"] = "mutated"
    recached = cache_module.get_cache(key)
    assert recached["research_data"]["status"] == "complete"


def test_expired_cache_entry_is_ignored(monkeypatch) -> None:
    key = cache_module.build_research_cache_key("ttl check")
    payload = {"research_data": {"status": "complete"}, "sources": []}

    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 1000)
    assert cache_module.set_cache(key, payload, ttl_seconds=10) is True

    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 1011)
    assert cache_module.get_cache(key) is None


def test_ttl_zero_does_not_expire_entry(monkeypatch) -> None:
    key = cache_module.build_research_cache_key("ttl zero")
    payload = {"research_data": {"status": "complete"}, "sources": []}

    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 1000)
    assert cache_module.set_cache(key, payload, ttl_seconds=0) is True
    monkeypatch.setattr(cache_module, "_now_epoch_seconds", lambda: 999_999)
    assert cache_module.get_cache(key) == payload


def test_delete_cache_removes_single_entry_only() -> None:
    key_a = cache_module.build_research_cache_key("delete a")
    key_b = cache_module.build_research_cache_key("delete b")
    payload = {"research_data": {"status": "complete"}, "sources": []}

    assert cache_module.set_cache(key_a, payload, ttl_seconds=60) is True
    assert cache_module.set_cache(key_b, payload, ttl_seconds=60) is True
    assert cache_module.delete_cache(key_a) is True
    assert cache_module.get_cache(key_a) is None
    assert cache_module.get_cache(key_b) == payload


def test_clear_cache_removes_entries() -> None:
    key_a = cache_module.build_research_cache_key("a")
    key_b = cache_module.build_research_cache_key("b")
    assert cache_module.set_cache(key_a, {"value": 1}, ttl_seconds=60) is True
    assert cache_module.set_cache(key_b, {"value": 2}, ttl_seconds=60) is True

    cache_module.clear_cache()
    assert cache_module.get_cache(key_a) is None
    assert cache_module.get_cache(key_b) is None


def test_set_and_get_cached_research_round_trip_updates_metadata() -> None:
    state = create_initial_state()
    key = cache_module.build_research_cache_key("edge computing")
    payload = {
        "research_data": {"status": "complete", "degraded": False},
        "sources": [{"title": "Edge Source", "url": "https://example.com/edge"}],
    }

    updates = cache_module.set_cached_research(state, key, payload)
    merged = dict(state)
    merged.update(updates)

    cached = cache_module.get_cached_research(merged, key)
    assert cached == payload
    assert key in merged["cache_metadata"]["keys"]
    assert merged["cache_metadata"]["backend"] == "in_memory"
    assert merged["cache_metadata"]["ttl_seconds"] == 1800


def test_disabled_cache_behavior_is_handled_by_agent_wrapper_not_backend() -> None:
    state = create_initial_state(
        cache_metadata={
            "enabled": False,
            "ttl_seconds": 1800,
            "backend": "in_memory",
            "keys": [],
        }
    )
    key = cache_module.build_research_cache_key("disabled check")
    payload = {"research_data": {"status": "complete"}, "sources": []}

    assert cache_module.set_cache(key, payload, ttl_seconds=60) is True
    assert cache_module.get_cache(key) == payload

    assert cache_module.get_cached_research(state, key) is None
    assert cache_module.set_cached_research(state, key, payload) == {}


def test_non_serializable_payload_is_not_cached() -> None:
    key = cache_module.build_research_cache_key("non-serializable")
    payload = {
        "research_data": {"status": "complete"},
        "sources": [{"title": "bad"}],
        "raw": {1, 2, 3},
    }
    state = create_initial_state()

    assert cache_module.set_cache(key, payload, ttl_seconds=60) is False
    assert cache_module.set_cached_research(state, key, payload) == {}
    assert cache_module.get_cache(key) is None
