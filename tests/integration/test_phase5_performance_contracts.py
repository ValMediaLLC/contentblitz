from __future__ import annotations

import importlib
import json
import socket
import threading
import time
from types import SimpleNamespace
from urllib import request as urllib_request

import pytest

from contentblitz.agents import image_agent as image_agent_module
from contentblitz.agents import research_agent as research_agent_module
from contentblitz.state import create_initial_state

generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
search_web_module = importlib.import_module("contentblitz.tools.search_web")
perplexity_module = importlib.import_module("contentblitz.tools.perplexity")


def _block_network(*_args, **_kwargs):
    raise AssertionError("Unexpected real network call attempted during test.")


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch) -> None:
    monkeypatch.setattr(socket, "create_connection", _block_network)
    monkeypatch.setattr(urllib_request, "urlopen", _block_network)
    monkeypatch.setattr(search_web_module.request, "urlopen", _block_network)
    monkeypatch.setattr(perplexity_module.request, "urlopen", _block_network)


def _cache_disabled_state(query: str) -> dict:
    return create_initial_state(
        user_query=query,
        cache_metadata={
            "enabled": False,
            "ttl_seconds": 1800,
            "backend": "in_memory",
            "keys": [],
        },
    )


def test_async_research_fanout_remains_deterministic_and_ordered(monkeypatch) -> None:
    inflight = {"current": 0, "max": 0}
    lock = threading.Lock()

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (model, metadata)
        assert agent_key == "research_agent"
        if "Generate 3-5 search queries" in prompt:
            return {"output": json.dumps(["q1", "q2", "q3"])}
        return {"output": "Synthesized summary for deterministic ordering checks."}

    def fake_search_web(query, depth="standard", timeout_seconds=None):
        _ = timeout_seconds
        assert depth == "standard"
        wait_by_query = {"q1": 0.03, "q2": 0.01, "q3": 0.02}
        with lock:
            inflight["current"] += 1
            inflight["max"] = max(inflight["max"], inflight["current"])
        try:
            time.sleep(wait_by_query.get(query, 0.01))
        finally:
            with lock:
                inflight["current"] -= 1
        return {
            "results": [
                {
                    "title": f"{query} title",
                    "url": f"https://example.com/{query}",
                    "snippet": (
                        "Long enough snippet for non-degraded synthesis quality."
                    ),
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(
        _cache_disabled_state("phase5 async deterministic research")
    )

    assert inflight["max"] >= 2
    assert updates["research_data"]["queries"] == ["q1", "q2", "q3"]
    assert [item["title"] for item in updates["sources"][:3]] == [
        "q1 title",
        "q2 title",
        "q3 title",
    ]


def test_research_timing_metadata_fields_are_present_and_safe(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (model, metadata)
        assert agent_key == "research_agent"
        if "Generate 3-5 search queries" in prompt:
            return {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "output": json.dumps(["q1", "q2", "q3"]),
            }
        return {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "output": "Structured synthesis summary.",
        }

    def fake_search_web(query, depth="standard", timeout_seconds=None):
        _ = (query, timeout_seconds)
        assert depth == "standard"
        return {
            "results": [
                {
                    "title": "Research source",
                    "url": "https://example.com/source",
                    "snippet": (
                        "Long enough snippet for non-degraded synthesis quality."
                    ),
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    updates = research_agent_module.research_agent_node(
        _cache_disabled_state("phase5 timing metadata coverage")
    )
    research_data = updates["research_data"]

    assert isinstance(research_data["provider_latency_total_ms"], int)
    assert isinstance(research_data["provider_latency_wall_ms"], int)
    assert research_data["provider_latency_total_ms"] >= 0
    assert research_data["provider_latency_wall_ms"] >= 0
    assert research_data["provider_call_count"] == 5
    assert research_data["provider_call_count_by_provider"]["anthropic"] == 2
    assert research_data["provider_call_count_by_provider"]["serp_api"] == 3
    assert "provider_latency_by_provider_ms" in research_data
    assert "provider_timeout_count" in research_data
    assert "provider_timeout_count_by_provider" in research_data
    assert "search_provider_wall_timeout_ms" in research_data
    assert "search_provider_wall_timeout_triggered" in research_data


def test_research_agent_uses_model_policy_defaults(monkeypatch) -> None:
    monkeypatch.setenv(
        "CONTENTBLITZ_TEXT_MODEL_RESEARCH_AGENT_DEFAULT",
        "gpt-5.4-mini",
    )
    requested_models: list[str] = []

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (prompt, metadata)
        assert agent_key == "research_agent"
        requested_models.append(str(model))
        if len(requested_models) == 1:
            return {"output": json.dumps(["q1", "q2", "q3"])}
        return {"output": "Policy-backed synthesis output."}

    def fake_search_web(query, depth="standard", timeout_seconds=None):
        _ = (query, timeout_seconds)
        assert depth == "standard"
        return {
            "results": [
                {
                    "title": "Policy source",
                    "url": "https://example.com/policy",
                    "snippet": (
                        "Long enough snippet for non-degraded synthesis quality."
                    ),
                }
            ]
        }

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    _ = research_agent_module.research_agent_node(
        _cache_disabled_state("phase5 model policy defaults")
    )

    assert requested_models
    assert requested_models == ["gpt-5.4-mini", "gpt-5.4-mini"]


def test_image_provider_fallback_chain_is_mockable(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv("CONTENTBLITZ_IMAGE_PROVIDER", "stability_ai")
    monkeypatch.setenv("CONTENTBLITZ_IMAGE_PROVIDER_FALLBACK", "fal_ai")
    monkeypatch.setenv("STABILITY_API_KEY", "stability-test")
    monkeypatch.setenv("FAL_API_KEY", "fal-test")

    def failing_primary_generate(**kwargs):
        _ = kwargs
        raise RuntimeError("primary unavailable")

    def fallback_generate(**kwargs):
        _ = kwargs
        return {"images": [{"url": "https://img.example/fallback.png"}]}

    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda _api_key: SimpleNamespace(generate=failing_primary_generate),
    )
    monkeypatch.setattr(
        generate_image_module,
        "_build_fal_client",
        lambda _api_key: SimpleNamespace(generate=fallback_generate),
    )

    result = generate_image_module.generate_image(prompt="Fallback image prompt")

    assert result.degraded is False
    assert result.provider == "fal_ai"
    assert result.fallback_provider_attempted is True
    assert result.fallback_provider_used is True
    assert result.provider_call_count >= 2
    assert result.provider_call_count_by_provider["stability_ai"] == 1
    assert result.provider_call_count_by_provider["fal_ai"] == 1


def test_image_tool_live_call_gate_prevents_provider_invocation(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "0")
    monkeypatch.setenv("STABILITY_API_KEY", "stability-test")
    monkeypatch.setenv("FAL_API_KEY", "fal-test")
    called = {"primary": 0, "fallback": 0}

    def fail_if_called(_api_key):
        called["primary"] += 1
        raise AssertionError(
            "Provider client should not be built when live calls are disabled."
        )

    def fail_if_called_fallback(_api_key):
        called["fallback"] += 1
        raise AssertionError(
            "Fallback client should not be built when live calls are disabled."
        )

    monkeypatch.setattr(
        generate_image_module, "_build_openai_client", fail_if_called
    )
    monkeypatch.setattr(
        generate_image_module, "_build_fal_client", fail_if_called_fallback
    )

    result = generate_image_module.generate_image(prompt="Do not call live providers")

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "live_calls_disabled"
    assert called == {"primary": 0, "fallback": 0}


def test_cost_control_counters_remain_agent_owned(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (model, metadata)
        assert agent_key == "research_agent"
        if "Generate 3-5 search queries" in prompt:
            return {
                "output": json.dumps(["q1", "q2", "q3"]),
                "usage": {"total_tokens": 0},
            }
        return {"output": "Summary output", "usage": {"total_tokens": 0}}

    def fake_search_web(query, depth="standard", timeout_seconds=None):
        _ = (query, timeout_seconds)
        assert depth == "standard"
        return {
            "results": [
                {
                    "title": "Counter source",
                    "url": "https://example.com/counter",
                    "snippet": (
                        "Long enough snippet for non-degraded synthesis quality."
                    ),
                }
            ],
            "search_queries_used_this_session": 9999,
        }

    monkeypatch.setattr(research_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)

    state = _cache_disabled_state("phase5 counters are agent owned")
    state["cost_controls"]["search_queries_used_this_session"] = 1
    updates = research_agent_module.research_agent_node(state)

    assert updates["cost_controls"]["search_queries_used_this_session"] == 4
    assert updates["cost_controls"]["search_queries_used_this_session"] != 9999


def test_degraded_image_provider_path_remains_recoverable(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (prompt, model, metadata)
        assert agent_key == "image_agent"
        return {"output": "Refined image prompt", "usage": {"total_tokens": 0}}

    def fake_generate_image(prompt, style="default"):
        _ = (prompt, style)
        return {
            "provider_used": "stability_ai",
            "provider_call_count": 1,
            "images": [],
            "error": {
                "code": "provider_unavailable",
                "message": "Image generation provider is unavailable.",
                "recoverable": True,
            },
        }

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)

    updates = image_agent_module.image_agent_node(
        create_initial_state(user_query="Create an image concept")
    )

    assert updates["draft_status"]["image"] == "failed"
    assert updates["tool_outputs"]["image_agent"]["recoverable"] is True
    assert updates["tool_outputs"]["image_agent"]["provider_status"] == "degraded"
    assert updates["errors"][-1]["recoverable"] is True
    assert any(
        "recoverable issue" in message.lower()
        for message in updates["status_messages"]
    )
