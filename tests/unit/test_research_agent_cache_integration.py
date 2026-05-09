from __future__ import annotations

import json

import pytest

from contentblitz.agents import research_agent as research_agent_module
from contentblitz.state import create_initial_state
from contentblitz.tools.cache import (
    build_research_cache_key,
    clear_cache,
    set_cache,
)


@pytest.fixture(autouse=True)
def _clear_process_cache() -> None:
    clear_cache()
    yield
    clear_cache()


def _mock_generate_text(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        assert agent_key == "research_agent"
        if "Generate 3-5 search queries" in prompt:
            return {"output": json.dumps(["q1", "q2", "q3"])}
        return {"output": "Synthesis summary."}

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)


def test_second_fresh_state_hits_shared_cache_and_skips_provider(monkeypatch) -> None:
    query = "ai governance"
    state_a = create_initial_state(user_query=query)
    _mock_generate_text(monkeypatch)

    calls = {"search": 0}

    def fake_search_web(query, depth="standard"):
        calls["search"] += 1
        return {
            "results": [
                {
                    "title": f"{query} source",
                    "url": f"https://provider.example/{depth}/{calls['search']}",
                    "snippet": "This snippet is long enough to be considered meaningful.",
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    first_updates = research_agent_module.research_agent_node(state_a)
    state_a.update(first_updates)

    expected_key = build_research_cache_key(query, depth="standard")
    assert calls["search"] > 0
    assert first_updates["research_data"]["cache_hit"] is False
    assert first_updates["research_data"]["degraded"] is False
    assert first_updates["cost_controls"]["search_queries_used_this_session"] > 0
    assert expected_key in first_updates["cache_metadata"]["keys"]

    state_b = create_initial_state(user_query=query)

    def fail_if_called(query, depth="standard"):
        raise AssertionError("search_web should not run when shared cache has a hit.")

    monkeypatch.setattr(research_agent_module, "search_web", fail_if_called)

    second_updates = research_agent_module.research_agent_node(state_b)
    state_b.update(second_updates)

    assert second_updates["research_data"]["cache_hit"] is True
    assert second_updates["research_data"]["degraded"] is False
    assert second_updates["research_data"]["status"] == "complete"
    assert second_updates["sources"] == first_updates["sources"]
    assert expected_key in second_updates["cache_metadata"]["keys"]
    assert state_b["cost_controls"]["search_queries_used_this_session"] == 0


def test_degraded_research_is_not_cached(monkeypatch) -> None:
    query = "nascent topic with no reliable sources"
    _mock_generate_text(monkeypatch)

    def degraded_search_web(query, depth="standard"):
        return {"results": []}

    state_a = create_initial_state(user_query=query)
    monkeypatch.setattr(research_agent_module, "search_web", degraded_search_web)
    first_updates = research_agent_module.research_agent_node(state_a)
    assert first_updates["research_data"]["degraded"] is True
    assert "cache_metadata" not in first_updates

    state_b = create_initial_state(user_query=query)
    calls = {"search": 0}

    def counting_degraded_search(query, depth="standard"):
        calls["search"] += 1
        return {"results": []}

    monkeypatch.setattr(research_agent_module, "search_web", counting_degraded_search)
    second_updates = research_agent_module.research_agent_node(state_b)

    assert calls["search"] > 0
    assert second_updates["research_data"]["cache_hit"] is False
    assert second_updates["research_data"]["degraded"] is True


def test_disabled_cache_bypasses_cache_reads_and_writes(monkeypatch) -> None:
    query = "ai risk controls"
    key = build_research_cache_key(query, depth="standard")
    cached_payload = {
        "research_data": {
            "status": "complete",
            "degraded": False,
            "quality": "standard",
            "synthesized_summary": "cached summary",
        },
        "sources": [{"title": "Cached", "url": "https://cached.example"}],
    }
    assert set_cache(key, cached_payload, ttl_seconds=600) is True

    _mock_generate_text(monkeypatch)

    calls = {"search": 0}

    def live_search_web(query, depth="standard"):
        calls["search"] += 1
        return {
            "results": [
                {
                    "title": "Live Source",
                    "url": "https://provider.example/live",
                    "snippet": "This snippet is long enough to be considered meaningful.",
                }
            ]
        }

    disabled_meta = {
        "enabled": False,
        "ttl_seconds": 1800,
        "backend": "in_memory",
        "keys": [],
    }

    state_a = create_initial_state(user_query=query, cache_metadata=disabled_meta)
    monkeypatch.setattr(research_agent_module, "search_web", live_search_web)
    first_updates = research_agent_module.research_agent_node(state_a)

    assert calls["search"] > 0
    assert first_updates["research_data"]["cache_hit"] is False
    assert "cache_metadata" not in first_updates

    state_b = create_initial_state(user_query=query, cache_metadata=disabled_meta)
    calls["search"] = 0
    monkeypatch.setattr(research_agent_module, "search_web", live_search_web)
    second_updates = research_agent_module.research_agent_node(state_b)

    assert calls["search"] > 0
    assert second_updates["research_data"]["cache_hit"] is False
