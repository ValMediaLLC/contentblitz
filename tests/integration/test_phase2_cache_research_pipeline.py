from __future__ import annotations

import json

import pytest

from contentblitz.agents import research_agent as research_agent_module
from contentblitz.state import create_initial_state
from contentblitz.tools.cache import clear_cache


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


def test_shared_cache_prevents_provider_call_for_fresh_state(monkeypatch) -> None:
    query = "latest AI content marketing trends"
    _mock_generate_text(monkeypatch)

    state_a = create_initial_state(user_query=query)
    run1_calls = {"search": 0}

    def run1_search_web(query, depth="standard"):
        run1_calls["search"] += 1
        return {
            "results": [
                {
                    "title": "AI trend source",
                    "url": "https://example.com/ai-trends",
                    "snippet": "This snippet is long enough to be considered meaningful.",
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "search_web", run1_search_web)
    run1_updates = research_agent_module.research_agent_node(state_a)
    state_a.update(run1_updates)

    assert run1_calls["search"] > 0
    assert run1_updates["research_data"]["cache_hit"] is False
    assert run1_updates["research_data"]["degraded"] is False

    state_b = create_initial_state(user_query=query)

    def fail_if_called(query, depth="standard"):
        raise AssertionError("search_web should not be called on cache hit.")

    monkeypatch.setattr(research_agent_module, "search_web", fail_if_called)
    run2_updates = research_agent_module.research_agent_node(state_b)
    state_b.update(run2_updates)

    assert run2_updates["research_data"]["cache_hit"] is True
    assert run2_updates["research_data"]["degraded"] is False
    assert run2_updates["sources"] == run1_updates["sources"]
    assert state_b["cost_controls"]["search_queries_used_this_session"] == 0
