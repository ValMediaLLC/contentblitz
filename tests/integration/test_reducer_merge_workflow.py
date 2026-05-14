from __future__ import annotations

import importlib
import socket
from types import SimpleNamespace
from urllib import request as urllib_request

import pytest

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
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


def _long_linkedin_post() -> str:
    paragraph = (
        "Reliable orchestration comes from deterministic state transitions, "
        "clear ownership, and bounded retries with visible recovery paths. "
    )
    return (
        "If your content workflow still feels unpredictable, start by stabilizing merge semantics.\n\n"
        + (paragraph * 9)
        + "\n\nWhat merge edge case did your team fix most recently?\n"
        "#AI #ContentOps #Engineering"
    )


def _text_payload_for_prompt(prompt: str) -> str:
    if "Create a JSON content brief for 'blog'" in prompt:
        return '{"format":"blog","objective":"educate","audience":"marketers","tone":"practical","angle":"systems"}'
    if "Create a JSON content brief for 'linkedin'" in prompt:
        return '{"format":"linkedin","objective":"engage","audience":"operators","tone":"direct","angle":"insight"}'
    if "Create a JSON content brief for 'image'" in prompt:
        return '{"format":"image","prompt_focus":"workflow orchestration dashboard","visual_direction":"clean, high contrast"}'
    if "Write an SEO-friendly blog draft in markdown." in prompt:
        return (
            "# Deterministic Merge Strategies\n\n"
            "Parallel fan-out can remain predictable when reducers are explicit, "
            "state ownership is isolated, and warning/error aggregation is normalized."
        )
    if "Write a LinkedIn post in plain text." in prompt:
        return _long_linkedin_post()
    if "Enhance this image generation prompt for clarity and visual detail." in prompt:
        return "Create a high-detail operations dashboard visual with cinematic lighting."
    return "Generic mocked provider response."


def _make_text_client(*, total_tokens: int = 12):
    def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        text = _text_payload_for_prompt(prompt)
        prompt_tokens = max(1, total_tokens // 2)
        completion_tokens = max(0, total_tokens - prompt_tokens)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _make_failing_image_client():
    calls: list[str] = []

    def generate(**kwargs):
        calls.append(kwargs["model"])
        raise RuntimeError("mocked image provider failure")

    client = SimpleNamespace(images=SimpleNamespace(generate=generate))
    return client, calls


def _preclassified_state(outputs: list[str]) -> dict:
    return create_initial_state(
        user_query="",
        requested_outputs=outputs,
        intent="content_creation",
        research_required=False,
    )


def test_parallel_fanout_preserves_text_outputs_when_image_branch_fails(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(generate_text_module, "_build_openai_client", lambda api_key: _make_text_client())
    image_client, calls = _make_failing_image_client()
    monkeypatch.setattr(generate_image_module, "_build_openai_client", lambda api_key: image_client)

    graph = build_langgraph()
    result = graph.invoke(_preclassified_state(["blog", "linkedin", "image"]))

    assert result["workflow_status"] == "partial_success"
    assert result["content_drafts"]["blog"]["body"].strip()
    assert result["content_drafts"]["linkedin"]["body"].strip()
    assert result["draft_status"]["blog"] == "complete"
    assert result["draft_status"]["linkedin"] == "complete"
    assert result["draft_status"]["image"] == "failed"
    assert any(item.get("recoverable") is True for item in result.get("errors", []))
    assert calls == ["dall-e-3", "dall-e-2"]
    assert result.get("final_response", "").strip()


def test_parallel_fanout_state_is_deterministic_across_repeated_runs(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(generate_text_module, "_build_openai_client", lambda api_key: _make_text_client())
    image_client, _ = _make_failing_image_client()
    monkeypatch.setattr(generate_image_module, "_build_openai_client", lambda api_key: image_client)

    graph = build_langgraph()
    first = graph.invoke(_preclassified_state(["blog", "linkedin", "image"]))
    second = graph.invoke(_preclassified_state(["blog", "linkedin", "image"]))

    assert first["workflow_status"] == second["workflow_status"] == "partial_success"
    assert first["content_drafts"]["blog"]["body"] == second["content_drafts"]["blog"]["body"]
    assert first["content_drafts"]["linkedin"]["body"] == second["content_drafts"]["linkedin"]["body"]
    assert first["draft_status"] == second["draft_status"]
    assert first["cost_controls"]["tokens_used_this_session"] == second["cost_controls"]["tokens_used_this_session"]
    assert first["cost_controls"]["budget_exceeded"] == second["cost_controls"]["budget_exceeded"]

