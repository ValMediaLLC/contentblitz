from __future__ import annotations

import importlib
import json
import socket
from types import SimpleNamespace
from urllib import request as urllib_request

import pytest

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
search_web_module = importlib.import_module("contentblitz.tools.search_web")
perplexity_module = importlib.import_module("contentblitz.tools.perplexity")


def _block_network(*args, **kwargs):
    raise AssertionError("Unexpected real network call attempted.")


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch) -> None:
    monkeypatch.setattr(socket, "create_connection", _block_network)
    monkeypatch.setattr(urllib_request, "urlopen", _block_network)
    monkeypatch.setattr(search_web_module.request, "urlopen", _block_network)
    monkeypatch.setattr(perplexity_module.request, "urlopen", _block_network)


def _make_text_client(response_resolver):
    def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        model = kwargs["model"]
        payload = response_resolver(prompt=prompt, model=model)
        text = payload["text"]
        total_tokens = int(payload.get("total_tokens", 8))
        prompt_tokens = max(1, total_tokens // 2)
        completion_tokens = max(0, total_tokens - prompt_tokens)
        return SimpleNamespace(
            model=model,
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _research_text_resolver(*, prompt: str, model: str) -> dict[str, object]:
    if "Generate 3-5 search queries as JSON list for this topic" in prompt:
        return {"text": json.dumps(["q1", "q2", "q3"]), "total_tokens": 6}
    if "Synthesize a concise research brief from these findings." in prompt:
        return {"text": "Phase 2 research synthesis summary.", "total_tokens": 9}
    return {"text": "generic", "total_tokens": 5}


def _research_only_state(
    *, query: str = "research phase2 query", cost_controls: dict | None = None
) -> dict:
    state = create_initial_state(
        user_query="",
        requested_outputs=["research"],
        research_required=True,
        intent="research",
        routing_decision="research_agent_node",
    )
    # Keep requested outputs pre-classified while letting research agent own the query text.
    state["user_query"] = query
    if cost_controls is not None:
        state["cost_controls"] = cost_controls
    return state


def test_research_only_with_serp_success_and_cache_miss_hit(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(_research_text_resolver),
    )

    call_count = {"serp": 0}

    def fake_http_get_json(_url: str):
        call_count["serp"] += 1
        return {
            "organic_results": [
                {
                    "title": "Canonical Source",
                    "link": "https://example.com/serp-source",
                    "snippet": "This source snippet is definitely meaningful and long enough.",
                    "source": "Example",
                    "date": "2026-05-09",
                }
            ]
        }

    monkeypatch.setattr(search_web_module, "_http_get_json", fake_http_get_json)

    graph = build_langgraph()
    state_a = _research_only_state(query="research latest ai content marketing trends")
    result_a = graph.invoke(state_a)

    assert result_a["final_response"].strip()
    assert result_a["workflow_status"] in {"success", "partial_success"}
    assert result_a["research_data"]["cache_hit"] is False
    assert call_count["serp"] > 0
    assert result_a["cost_controls"]["search_queries_used_this_session"] > 0
    # dedupe across repeated query executions keeps one URL source.
    urls = [src.get("url") for src in result_a["sources"] if src.get("url")]
    assert len(urls) == len(set(urls))
    assert len(result_a["sources"]) == 1
    assert "(https://example.com/serp-source)" in result_a["final_response"]

    # Cache hit should skip provider calls on a fresh state.
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: (_ for _ in ()).throw(
            AssertionError("SERP should not be called on cache hit.")
        ),
    )
    state_b = _research_only_state(query="research latest ai content marketing trends")
    result_b = graph.invoke(state_b)

    assert result_b["research_data"]["cache_hit"] is True
    assert result_b["cost_controls"]["search_queries_used_this_session"] == 0
    assert result_b["final_response"].strip()
    assert result_b["workflow_status"] in {"success", "partial_success"}


def test_research_only_serp_degraded_perplexity_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "px-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(_research_text_resolver),
    )

    # SERP returns too-short snippet, forcing fallback.
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: {
            "organic_results": [
                {
                    "title": "Weak SERP",
                    "link": "https://example.com/weak",
                    "snippet": "tiny",
                    "source": "Example",
                }
            ]
        },
    )
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": "Perplexity fallback produced usable research text for synthesis.",
                    },
                    "citations": [],
                }
            ]
        },
    )

    graph = build_langgraph()
    result = graph.invoke(_research_only_state(query="research fallback scenario"))

    assert result["final_response"].strip()
    assert result["research_data"]["fallback_used"] is True
    assert any(source.get("provider") == "perplexity" for source in result["sources"])
    # Perplexity with missing URL must not expose fake citations.
    for source in result["sources"]:
        if source.get("provider") == "perplexity" and source.get("url") is None:
            assert source["citation_available"] is False
    # No fake URL formatting should appear for missing citation entries.
    assert "(None)" not in result["final_response"]


def test_search_cap_reached_produces_partial_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(_research_text_resolver),
    )
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: (_ for _ in ()).throw(
            AssertionError("SERP should not run when search cap is reached.")
        ),
    )

    graph = build_langgraph()
    result = graph.invoke(
        _research_only_state(
            query="research search cap scenario",
            cost_controls={
                "tokens_used_this_session": 0,
                "search_queries_used_this_session": 5,
                "image_generations_used_this_session": 0,
                "total_retries_used_this_session": 0,
                "budget_exceeded": False,
                "search_query_cap_per_session": 5,
            },
        )
    )

    assert result["research_data"]["degraded"] is True
    assert result["research_data"]["degraded_reason"] == "search_cap_reached"
    assert result["workflow_status"] == "partial_success"
    assert result["final_response"].strip()
