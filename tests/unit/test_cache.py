from contentblitz.state import create_initial_state
from contentblitz.tools.cache import (
    build_research_cache_key,
    get_cached_research,
    normalize_query,
    set_cached_research,
)


def test_normalize_query_is_deterministic() -> None:
    assert normalize_query("  AI   Trends  2026 ") == "ai trends 2026"


def test_build_research_cache_key_follows_spec_shape() -> None:
    key = build_research_cache_key("AI market outlook", depth="deep")
    assert key.startswith("research:")
    assert key.endswith(":deep")
    assert "AI market outlook" not in key


def test_set_and_get_cached_research_round_trip() -> None:
    state = create_initial_state()
    key = build_research_cache_key("cloud security")
    payload = {
        "research_data": {"status": "complete", "degraded": False},
        "sources": [{"title": "A", "url": "https://example.com"}],
    }

    updates = set_cached_research(state, key, payload)
    merged = dict(state)
    merged.update(updates)

    cached = get_cached_research(merged, key)
    assert cached == payload
    assert key in merged["cache_metadata"]["keys"]


def test_get_cached_research_respects_disabled_cache() -> None:
    state = create_initial_state(cache_metadata={"enabled": False, "ttl_seconds": 1800, "backend": "in_memory", "keys": []})
    key = build_research_cache_key("edge computing")
    assert get_cached_research(state, key) is None

