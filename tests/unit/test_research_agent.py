import json

from contentblitz.agents import research_agent as research_agent_module
from contentblitz.state import create_initial_state
from contentblitz.tools.cache import build_research_cache_key, set_cached_research


def _assert_complete_research_data(payload: dict) -> None:
    data = payload["research_data"]
    assert isinstance(data.get("synthesized_summary"), str)
    assert data["synthesized_summary"].strip()
    assert data.get("quality") in {"standard", "degraded"}
    assert isinstance(data.get("key_facts"), list)
    assert len(data["key_facts"]) >= 3
    assert all(isinstance(item, str) and item.strip() for item in data["key_facts"])
    assert isinstance(data.get("keywords"), list)
    assert len(data["keywords"]) >= 3
    assert all(isinstance(item, str) and item.strip() for item in data["keywords"])
    assert isinstance(data.get("entities"), list)
    for source in payload.get("sources", []):
        assert isinstance(source.get("snippet"), str)
        assert source["snippet"].strip()


def _mock_generate_text(monkeypatch):
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        assert agent_key == "research_agent"
        if "Generate 3-5 search queries" in prompt:
            return {"output": json.dumps(["q1", "q2", "q3"])}
        return {"output": "Synthesis summary."}

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)


def test_cache_hit_skips_search_web(monkeypatch) -> None:
    key = build_research_cache_key("topic")
    state = create_initial_state(user_query="topic")
    payload = {
        "research_data": {
            "status": "complete",
            "degraded": False,
            "quality": "standard",
            "synthesized_summary": "cached",
            "summary": "cached",
            "key_facts": ["f1", "f2", "f3"],
            "keywords": ["k1", "k2", "k3"],
            "entities": ["E1"],
        },
        "sources": [{"title": "Cached", "url": "https://cached.example"}],
    }
    cache_updates = set_cached_research(state, key, payload)
    state.update(cache_updates)

    calls = {"search": 0}

    def fail_if_called(query, depth="standard"):
        calls["search"] += 1
        raise AssertionError("search_web should not be called on cache hit")

    monkeypatch.setattr(research_agent_module, "search_web", fail_if_called)
    _mock_generate_text(monkeypatch)

    updates = research_agent_module.research_agent_node(state)
    assert calls["search"] == 0
    assert updates["research_data"]["cache_hit"] is True
    assert updates["sources"][0]["title"] == "Cached"
    _assert_complete_research_data(updates)


def test_cache_miss_calls_search_web(monkeypatch) -> None:
    state = create_initial_state(user_query="zero trust architecture")
    _mock_generate_text(monkeypatch)

    calls = {"search": 0}

    def fake_search_web(query, depth="standard"):
        calls["search"] += 1
        return {
            "results": [
                {
                    "title": f"{query} source",
                    "url": "https://example.com/source",
                    "snippet": (
                        "This is a long enough snippet to be treated "
                        "as non-degraded."
                    ),
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(state)
    assert calls["search"] > 0
    assert "research_data" in updates
    assert isinstance(updates["sources"], list)
    assert updates["cost_controls"]["search_queries_used_this_session"] > 0
    _assert_complete_research_data(updates)


def test_degraded_snippets_trigger_fallback(monkeypatch) -> None:
    state = create_initial_state(user_query="supply chain resilience")
    _mock_generate_text(monkeypatch)

    depths = []

    def fake_search_web(query, depth="standard"):
        depths.append(depth)
        if depth == "standard":
            return {
                "results": [
                    {
                        "title": "SERP",
                        "url": "https://serp.example",
                        "snippet": "short",
                    },
                ]
            }
        return {
            "results": [
                {
                    "title": "Perplexity",
                    "url": "https://px.example",
                    "snippet": (
                        "This fallback snippet is long enough "
                        "for research synthesis."
                    ),
                    "citation_available": True,
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(state)
    assert "fallback" in depths
    assert updates["research_data"]["fallback_used"] is True
    _assert_complete_research_data(updates)


def test_perplexity_source_shape_when_fallback_has_no_results(monkeypatch) -> None:
    state = create_initial_state(user_query="distributed systems latency")
    _mock_generate_text(monkeypatch)

    def fake_search_web(query, depth="standard"):
        if depth == "standard":
            return {
                "results": [
                    {"title": "SERP", "url": "https://serp.example", "snippet": "tiny"}
                ]
            }
        return {"results": []}

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)
    updates = research_agent_module.research_agent_node(state)

    perplexity_sources = [
        item for item in updates["sources"] if item.get("provider") == "perplexity"
    ]
    assert len(perplexity_sources) >= 1
    assert perplexity_sources[0]["url"] is None
    assert perplexity_sources[0]["citation_available"] is False
    assert perplexity_sources[0]["snippet"].strip()
    _assert_complete_research_data(updates)


def test_search_query_cap_prevents_extra_calls(monkeypatch) -> None:
    state = create_initial_state(
        user_query="ai policy",
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 5,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "search_query_cap_per_session": 5,
        },
    )
    _mock_generate_text(monkeypatch)

    calls = {"search": 0}

    def fake_search_web(query, depth="standard"):
        calls["search"] += 1
        return {"results": []}

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(state)
    assert calls["search"] == 0
    assert updates["research_data"]["degraded"] is True
    assert updates["research_data"]["quality"] == "degraded"
    _assert_complete_research_data(updates)


def test_failed_providers_produce_degraded_research_data(monkeypatch) -> None:
    state = create_initial_state(user_query="renewable grid stability")
    _mock_generate_text(monkeypatch)

    def failing_search_web(query, depth="standard"):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(research_agent_module, "search_web", failing_search_web)
    updates = research_agent_module.research_agent_node(state)

    assert updates["research_data"]["degraded"] is True
    assert updates["research_data"]["status"] == "degraded"
    assert updates["research_data"]["quality"] == "degraded"
    _assert_complete_research_data(updates)


def test_cache_does_not_store_degraded_results(monkeypatch) -> None:
    state = create_initial_state(user_query="nascent topic with weak citations")
    _mock_generate_text(monkeypatch)

    def fake_search_web(query, depth="standard"):
        return {"results": []}

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)
    updates = research_agent_module.research_agent_node(state)

    assert updates["research_data"]["degraded"] is True
    assert "tool_outputs" not in updates
    assert "cache_metadata" not in updates
    _assert_complete_research_data(updates)


def test_research_agent_updates_only_allowed_state_fields(monkeypatch) -> None:
    state = create_initial_state(user_query="AI SEO LinkedIn marketing trends")
    _mock_generate_text(monkeypatch)

    def fake_search_web(query, depth="standard"):
        return {
            "results": [
                {
                    "title": "AI SEO LinkedIn marketing",
                    "url": "https://example.com/source",
                    "snippet": (
                        "This snippet is long enough "
                        "for deterministic synthesis."
                    ),
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)
    updates = research_agent_module.research_agent_node(state)

    allowed_keys = {
        "research_data",
        "sources",
        "cost_controls",
        "workflow_status",
        "final_response",
        "cache_metadata",
    }
    assert set(updates.keys()).issubset(allowed_keys)
    assert "requested_outputs" not in updates
    assert "routing_decision" not in updates


def test_deterministic_research_summary_used_when_text_synthesis_unavailable(
    monkeypatch,
) -> None:
    state = create_initial_state(user_query="best electric cars to buy in 2026")

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        if "Generate 3-5 search queries" in prompt:
            return {"output": json.dumps(["best electric cars 2026 analysis"])}
        return {
            "output": "",
            "degraded": True,
            "error": {"code": "authentication_failed", "recoverable": True},
        }

    def fake_search_web(query, depth="standard"):
        return {
            "results": [
                {
                    "title": "IEA EV Outlook 2026",
                    "url": "https://www.iea.org/reports/global-ev-outlook-2026",
                    "snippet": (
                        "EV adoption growth is accelerating while charging "
                        "infrastructure, pricing, and range remain key buyer factors."
                    ),
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(state)
    summary = updates["research_data"]["synthesized_summary"]

    assert updates["research_data"]["deterministic_summary_used"] is True
    assert "## Research Summary" in summary
    assert "Sources reviewed: 1" in summary
    assert "Citation-ready sources: 1" in summary
    assert "iea.org" in summary
    assert "Retrieved Themes" in summary
    assert "Useful Source Leads" in summary
    assert "IEA EV Outlook 2026" in summary


def test_deterministic_research_summary_handles_no_sources_safely() -> None:
    summary = research_agent_module._deterministic_research_summary(  # noqa: SLF001
        query="emerging AI market",
        sources=[],
    )

    assert "no usable sources were retrieved" in summary.lower()
