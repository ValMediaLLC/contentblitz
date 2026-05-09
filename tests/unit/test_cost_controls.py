from __future__ import annotations

import json

from contentblitz.agents import (
    blog_writer as blog_writer_module,
    image_agent as image_agent_module,
    query_handler as query_handler_module,
    research_agent as research_agent_module,
    retry_router as retry_router_module,
)
from contentblitz.core.cost_controls import (
    apply_text_tokens,
    normalize_cost_controls,
    preferred_text_model,
    token_budget_exceeded,
)
from contentblitz.state import create_initial_state
from contentblitz.tools import image as image_tool
from contentblitz.tools import text as text_tool
from contentblitz.tools import web_search as web_search_tool
from contentblitz.tools.generate_image import GenerateImageResult
from contentblitz.tools.generate_text import GenerateTextResult
from contentblitz.tools.provider_types import SearchResult, SearchWebResult


def test_token_counter_increments_from_generate_text_total_tokens() -> None:
    controls = normalize_cost_controls({"tokens_used_this_session": 7})
    updated = apply_text_tokens(controls, {"usage": {"total_tokens": 13}})
    assert updated["tokens_used_this_session"] == 20


def test_near_token_cap_prefers_gpt4o_mini() -> None:
    controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 900,
            "token_budget_per_session": 1000,
        }
    )
    assert preferred_text_model(controls) == "gpt-4o-mini"


def test_exceeded_token_cap_sets_budget_exceeded_true(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"intent": "content_creation", "requested_outputs": ["blog"]}), "usage": {"total_tokens": 30}}

    monkeypatch.setattr(query_handler_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        user_query="write a blog",
        cost_controls={
            "tokens_used_this_session": 90,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 100,
        },
    )
    updates = query_handler_module.query_handler_node(state)

    assert updates["routing_decision"] == "error_handler_node"
    assert updates["cost_controls"]["budget_exceeded"] is True
    assert token_budget_exceeded(updates["cost_controls"]) is True


def test_search_cap_prevents_additional_search_calls(monkeypatch) -> None:
    calls = {"search": 0}

    def fake_search_web(query, depth="standard"):
        calls["search"] += 1
        return {"results": []}

    monkeypatch.setattr(research_agent_module, "search_web", fake_search_web)
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
    updates = research_agent_module.research_agent_node(state)

    assert calls["search"] == 0
    assert updates["research_data"]["degraded"] is True
    assert updates["research_data"]["degraded_reason"] == "search_cap_reached"


def test_image_cap_prevents_generation_and_writes_recoverable_error(monkeypatch) -> None:
    calls = {"image": 0}

    def fake_generate_image(prompt, style="default"):
        calls["image"] += 1
        return {"images": [{"url": "https://example.com/image.png"}]}

    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    state = create_initial_state(
        user_query="Create image",
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 2,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "image_generation_cap_per_session": 2,
        },
    )
    updates = image_agent_module.image_agent_node(state)

    assert calls["image"] == 0
    assert updates["tool_outputs"]["image_agent"]["status"] == "skipped"
    assert updates["errors"][-1]["recoverable"] is True
    assert updates["errors"][-1]["type"] == "image_generation_cap_reached"


def test_retry_cap_prevents_retry_fan_out() -> None:
    state = create_initial_state(
        retry_requested=True,
        quality_scores={"blog": {"validation_status": "retry_needed", "composite": 0.6}},
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 1,
            "max_total_retries_per_session": 1,
            "budget_exceeded": False,
        },
    )
    updates = retry_router_module.retry_router_node(state)

    assert updates["retry_requested"] is False
    assert updates["retry_targets"] == []
    assert updates["_retry_counts_incremented"] is False


def test_tools_do_not_mutate_cost_counters(monkeypatch) -> None:
    counters = {
        "tokens_used_this_session": 9,
        "search_queries_used_this_session": 2,
        "image_generations_used_this_session": 1,
        "total_retries_used_this_session": 1,
        "budget_exceeded": False,
    }
    original = dict(counters)

    monkeypatch.setattr(
        text_tool,
        "_core_generate_text",
        lambda **kwargs: GenerateTextResult(
            text="ok",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
            degraded=False,
            error=None,
        ),
    )
    monkeypatch.setattr(
        web_search_tool,
        "_core_search_web",
        lambda **kwargs: SearchWebResult(
            provider="serp",
            query=kwargs.get("query", ""),
            results=[
                SearchResult(
                    title="t",
                    url="https://example.com",
                    snippet="s",
                    source="serp",
                    published_at=None,
                    citation_available=True,
                    credibility_score=0.8,
                )
            ],
            degraded=False,
            error=None,
        ),
    )
    monkeypatch.setattr(
        image_tool,
        "_core_generate_image",
        lambda **kwargs: GenerateImageResult(
            provider="openai",
            model="dall-e-3",
            prompt=kwargs.get("prompt", ""),
            image_url="https://example.com/img.png",
            revised_prompt=None,
            degraded=False,
            error=None,
        ),
    )

    text_result = text_tool.generate_text(prompt="hello", agent_key="query_handler")
    search_result = web_search_tool.search_web(query="ai trends")
    image_result = image_tool.generate_image(prompt="hero image")

    assert "cost_controls" not in text_result
    assert "cost_controls" not in search_result
    assert "cost_controls" not in image_result
    assert counters == original
