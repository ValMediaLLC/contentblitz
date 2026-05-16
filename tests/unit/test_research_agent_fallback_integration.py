from __future__ import annotations

import json

from contentblitz.agents import research_agent as research_agent_module
from contentblitz.state import create_initial_state


def _mock_generate_text(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        assert agent_key == "research_agent"
        if "Generate 3-5 search queries" in prompt:
            return {"output": json.dumps(["q1", "q2", "q3"])}
        return {"output": "Synthesis summary from fallback flow."}

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)


def test_research_agent_fallback_writes_perplexity_source_metadata_and_updates_counter(
    monkeypatch,
) -> None:
    state = create_initial_state(
        user_query="ai workflow trends",
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "search_query_cap_per_session": 2,
        },
    )
    _mock_generate_text(monkeypatch)

    calls = []

    def fake_search_web(query, depth="standard"):
        calls.append(depth)
        if depth == "standard":
            return {
                "results": [
                    {
                        "title": "SERP weak",
                        "url": "https://serp.example/weak",
                        "snippet": "tiny",
                    }
                ]
            }
        return {
            "results": [
                {
                    "title": "Perplexity fallback",
                    "url": None,
                    "snippet": "This fallback snippet is long and usable for synthesis.",
                    "citation_available": False,
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(state)
    assert calls == ["standard", "fallback"]
    assert updates["research_data"]["fallback_used"] is True
    assert updates["cost_controls"]["search_queries_used_this_session"] == 2

    sources = updates["sources"]
    assert len(sources) >= 1
    perplexity_sources = [s for s in sources if s.get("provider") == "perplexity"]
    assert perplexity_sources
    first = perplexity_sources[0]
    assert first["url"] is None
    assert first["citation_available"] is False
    assert isinstance(first["snippet"], str) and first["snippet"].strip()


def test_research_agent_does_not_cache_degraded_perplexity_only_fallback(
    monkeypatch,
) -> None:
    state = create_initial_state(
        user_query="nascent topic",
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "search_query_cap_per_session": 2,
        },
    )
    _mock_generate_text(monkeypatch)

    def fake_search_web(query, depth="standard"):
        if depth == "standard":
            return {
                "results": [
                    {
                        "title": "SERP weak",
                        "url": "https://serp.example/weak",
                        "snippet": "tiny",
                    }
                ]
            }
        return {"results": []}

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(state)
    assert updates["research_data"]["fallback_used"] is True
    assert updates["research_data"]["degraded"] is True
    assert updates["research_data"]["quality"] == "degraded"
    assert "tool_outputs" not in updates
    assert "cache_metadata" not in updates
