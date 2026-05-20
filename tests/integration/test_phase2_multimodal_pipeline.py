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
        "Most teams are not blocked by ideas. They are blocked by repeatability. "
        "A single prompt creates momentum, but systems create dependable output. "
    )
    return (
        "If your AI content workflow still feels random, this is the fix.\n\n"
        + (paragraph * 8)
        + "\n\nWhat process stage would you standardize first? Share below.\n"
        "#AI #MarketingOps #ContentStrategy"
    )


def _text_payload_for_prompt(prompt: str) -> str:
    if "Create a JSON content brief for 'blog'" in prompt:
        return json.dumps(
            {
                "format": "blog",
                "objective": "Educate marketers with clear implementation guidance.",
                "audience": "marketing leaders",
                "tone": "practical",
                "angle": "systems playbook",
            }
        )
    if "Create a JSON content brief for 'linkedin'" in prompt:
        return json.dumps(
            {
                "format": "linkedin",
                "objective": "Drive discussion among operators.",
                "audience": "marketing operators",
                "tone": "direct",
                "angle": "operational insight",
            }
        )
    if "Create a JSON content brief for 'image'" in prompt:
        return json.dumps(
            {
                "format": "image",
                "prompt_focus": "futuristic marketing operations control room",
                "visual_direction": "high contrast cinematic",
            }
        )
    if "Write an SEO-friendly blog draft in markdown." in prompt:
        return (
            "# AI Workflow Playbook\n\n"
            "Teams that define repeatable content workflows outperform teams "
            "that chase one-off prompts. This draft explains the structure, "
            "checkpoints, and ownership model needed for predictable output."
        )
    if "Write a LinkedIn post in plain text." in prompt:
        return _long_linkedin_post()
    if "Enhance this image generation prompt for clarity and visual detail." in prompt:
        return (
            "Create a high-detail futuristic marketing command center with rich "
            "cinematic lighting."
        )
    return "Generic mocked provider response."


def _make_text_client(*, total_tokens: int = 10):
    def create(**kwargs):
        model = kwargs["model"]
        prompt = kwargs["messages"][0]["content"]
        text = _text_payload_for_prompt(prompt)
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


def _make_image_client(*, fallback_to_dalle2: bool = False):
    calls = {"models": []}

    def generate(**kwargs):
        model = kwargs["model"]
        calls["models"].append(model)
        if fallback_to_dalle2 and model == "dall-e-3":
            raise RuntimeError("primary image model failed")
        return SimpleNamespace(
            data=[SimpleNamespace(url="https://img.example/phase2-image.png")]
        )

    return SimpleNamespace(images=SimpleNamespace(generate=generate)), calls


def _preclassified_state(
    outputs: list[str],
    *,
    research_required: bool = False,
    cost_controls: dict | None = None,
) -> dict:
    state = create_initial_state(
        user_query="",
        requested_outputs=outputs,
        research_required=research_required,
        intent="content_creation" if outputs != ["image"] else "image_generation",
    )
    if cost_controls is not None:
        state["cost_controls"] = cost_controls
    return state


def test_blog_only_with_openai_text_generation(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
    graph = build_langgraph()

    result = graph.invoke(_preclassified_state(["blog"]))

    assert result["workflow_status"] == "success"
    assert result["final_response"].strip()
    assert "## Blog Draft" in result["final_response"]
    assert result["content_drafts"]["blog"]["body"].strip()


def test_linkedin_only_with_openai_text_generation(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
    graph = build_langgraph()

    result = graph.invoke(_preclassified_state(["linkedin"]))

    assert result["workflow_status"] == "success"
    assert result["final_response"].strip()
    assert "## LinkedIn Draft" in result["final_response"]
    assert result["content_drafts"]["linkedin"]["body"].strip()


def test_blog_and_linkedin_token_cost_tracking(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(total_tokens=10),
    )
    graph = build_langgraph()

    result = graph.invoke(_preclassified_state(["blog", "linkedin"]))
    # Current deterministic merge behavior records 30 tokens in this fan-out path.
    assert result["cost_controls"]["tokens_used_this_session"] == 30
    assert result["cost_controls"]["search_queries_used_this_session"] == 0
    assert result["cost_controls"]["image_generations_used_this_session"] == 0
    assert result["final_response"].strip()
    assert result["workflow_status"] == "success"


def test_image_only_dalle3_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
    client, calls = _make_image_client(fallback_to_dalle2=False)
    monkeypatch.setattr(
        generate_image_module, "_build_openai_client", lambda api_key: client
    )
    graph = build_langgraph()

    result = graph.invoke(_preclassified_state(["image"], research_required=False))

    assert result["workflow_status"] == "success"
    assert result["final_response"].strip()
    assert calls["models"] == ["dall-e-3"]
    assert result["cost_controls"]["image_generations_used_this_session"] == 1
    for output in result.get("image_outputs", []):
        assert "base64" not in output
        assert "b64_json" not in output


def test_image_only_dalle3_fallback_to_dalle2(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
    client, calls = _make_image_client(fallback_to_dalle2=True)
    monkeypatch.setattr(
        generate_image_module, "_build_openai_client", lambda api_key: client
    )
    graph = build_langgraph()

    result = graph.invoke(_preclassified_state(["image"], research_required=False))

    assert result["workflow_status"] == "success"
    assert result["final_response"].strip()
    assert calls["models"] == ["dall-e-3", "dall-e-2"]
    assert any(
        item.get("provider") == "dall-e-2" for item in result.get("image_outputs", [])
    )


def test_blog_linkedin_image_all_mocked_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
    client, _ = _make_image_client(fallback_to_dalle2=False)
    monkeypatch.setattr(
        generate_image_module, "_build_openai_client", lambda api_key: client
    )
    graph = build_langgraph()

    result = graph.invoke(_preclassified_state(["blog", "linkedin", "image"]))

    assert result["workflow_status"] == "success"
    assert result["final_response"].strip()
    assert "## Blog Draft" in result["final_response"]
    assert "## LinkedIn Draft" in result["final_response"]
    assert "## Image Assets" in result["final_response"]
    for output in result.get("image_outputs", []):
        assert "base64" not in output
        assert "b64_json" not in output


def test_image_cap_reached_produces_partial_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
    client, _ = _make_image_client(fallback_to_dalle2=False)
    monkeypatch.setattr(
        generate_image_module, "_build_openai_client", lambda api_key: client
    )
    graph = build_langgraph()

    result = graph.invoke(
        _preclassified_state(
            ["blog", "image"],
            cost_controls={
                "tokens_used_this_session": 0,
                "search_queries_used_this_session": 0,
                "image_generations_used_this_session": 2,
                "total_retries_used_this_session": 0,
                "budget_exceeded": False,
                "image_generation_cap_per_session": 2,
            },
        )
    )

    assert result["workflow_status"] == "partial_success"
    assert result["final_response"].strip()
    assert "## Blog Draft" in result["final_response"]
    assert "recoverable issue" in result["final_response"].lower()


def test_token_budget_exceeded_safe_failure(monkeypatch) -> None:
    graph = build_langgraph()
    state = create_initial_state(
        user_query="write a blog post about ai workflows",
        cost_controls={
            "tokens_used_this_session": 1200,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": True,
            "token_budget_per_session": 1000,
        },
    )

    result = graph.invoke(state)
    assert result["workflow_status"] == "failed"
    assert result["final_response"].strip()
