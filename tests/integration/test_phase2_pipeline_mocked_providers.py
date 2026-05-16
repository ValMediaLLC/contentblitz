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
    segment = (
        "Execution velocity improves when content planning, research, and review are systemized. "
        "The goal is repeatability, not one-off output spikes. "
    )
    return (
        "Most AI content bottlenecks are process bottlenecks.\n\n"
        + (segment * 8)
        + "\n\nWhich step in your workflow creates the most drag today?\n"
        "#AI #ContentOps #MarketingStrategy"
    )


def _text_payload_for_prompt(prompt: str) -> str:
    if "Classify the user request for ContentBlitz." in prompt:
        return json.dumps(
            {
                "intent": "content_creation",
                "requested_outputs": ["blog", "linkedin", "image"],
                "research_required": True,
                "clarification_needed": False,
                "clarification_message": None,
                "export_requested": False,
            }
        )
    if "Generate 3-5 search queries as JSON list for this topic" in prompt:
        return json.dumps(
            [
                "latest ai content workflow trends",
                "ai marketing ops benchmarks",
                "ai content production case studies",
            ]
        )
    if "Synthesize a concise research brief from these findings." in prompt:
        return "Research synthesis produced from mocked provider results."
    if "Create a JSON content brief for 'blog'" in prompt:
        return json.dumps(
            {
                "format": "blog",
                "objective": "Educate operators.",
                "audience": "marketing teams",
                "tone": "practical",
                "angle": "systems execution",
            }
        )
    if "Create a JSON content brief for 'linkedin'" in prompt:
        return json.dumps(
            {
                "format": "linkedin",
                "objective": "Drive discussion.",
                "audience": "operators",
                "tone": "direct",
                "angle": "execution insight",
            }
        )
    if "Create a JSON content brief for 'image'" in prompt:
        return json.dumps(
            {
                "format": "image",
                "prompt_focus": "modern content operations command center",
                "visual_direction": "cinematic contrast",
            }
        )
    if "Write an SEO-friendly blog draft in markdown." in prompt:
        return (
            "# AI Workflow Systems\n\n"
            "A durable content workflow combines repeatable research inputs, scoped briefs, "
            "and measurable editorial standards."
        )
    if "Write a LinkedIn post in plain text." in prompt:
        return _long_linkedin_post()
    if "Enhance this image generation prompt for clarity and visual detail." in prompt:
        return (
            "Create a cinematic, high-detail content operations command center scene."
        )
    return "Generic mocked text response."


def _make_text_client(*, total_tokens: int = 8):
    def create(**kwargs):
        text = _text_payload_for_prompt(kwargs["messages"][0]["content"])
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

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _make_image_client():
    def generate(**kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(url="https://img.example/phase2-pipeline.png")]
        )

    return SimpleNamespace(images=SimpleNamespace(generate=generate))


def test_research_only_with_serp_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )

    calls = {"serp": 0, "perplexity": 0}

    def fake_serp(_url: str):
        calls["serp"] += 1
        return {
            "organic_results": [
                {
                    "title": "Primary research source",
                    "link": "https://example.com/research-source",
                    "snippet": "This source has meaningful details about AI workflow adoption patterns.",
                    "source": "Example",
                    "date": "2026-05-09",
                }
            ]
        }

    def fail_perplexity(**kwargs):
        calls["perplexity"] += 1
        raise AssertionError("Perplexity should not be called when SERP is usable.")

    monkeypatch.setattr(search_web_module, "_http_get_json", fake_serp)
    monkeypatch.setattr(perplexity_module, "_http_post_json", fail_perplexity)

    state = create_initial_state(
        user_query="research latest ai content workflow trends",
        requested_outputs=["research"],
        research_required=True,
        intent="research",
    )
    result = build_langgraph().invoke(state)

    assert result["final_response"].strip()
    assert result["workflow_status"] in {"success", "partial_success"}
    assert result["research_data"]["degraded"] is False
    assert result["research_data"]["fallback_used"] is False
    assert calls["serp"] > 0
    assert calls["perplexity"] == 0


def test_research_only_with_serp_degraded_perplexity_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "px-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )

    calls = {"perplexity": 0}

    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: {
            "organic_results": [
                {
                    "title": "Weak SERP source",
                    "link": "https://example.com/weak",
                    "snippet": "short",
                    "source": "Example",
                }
            ]
        },
    )

    def fake_perplexity(**kwargs):
        calls["perplexity"] += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": "Perplexity fallback provided usable context."
                    },
                    "citations": [],
                }
            ]
        }

    monkeypatch.setattr(perplexity_module, "_http_post_json", fake_perplexity)

    state = create_initial_state(
        user_query="research ai workflow fallback path",
        requested_outputs=["research"],
        research_required=True,
        intent="research",
    )
    result = build_langgraph().invoke(state)

    assert result["final_response"].strip()
    assert result["workflow_status"] in {"success", "partial_success"}
    assert result["research_data"]["fallback_used"] is True
    assert calls["perplexity"] > 0
    assert "(None)" not in result["final_response"]
    for source in result["sources"]:
        if source.get("provider") == "perplexity":
            if source.get("url") is None:
                assert source["citation_available"] is False


def test_blog_linkedin_image_with_all_providers_mocked_success(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: _make_image_client(),
    )

    calls = {"serp": 0}

    def fake_serp(_url: str):
        calls["serp"] += 1
        return {
            "organic_results": [
                {
                    "title": "Deduped source A",
                    "link": "https://example.com/source-a",
                    "snippet": "Detailed source snippet for meaningful synthesis and citation.",
                    "source": "ExampleA",
                },
                {
                    "title": "Duplicate source A",
                    "link": "https://example.com/source-a",
                    "snippet": "Duplicate URL should be removed.",
                    "source": "ExampleA",
                },
                {
                    "title": "No URL source",
                    "snippet": "Still useful source text but without a URL.",
                    "source": "ExampleB",
                },
            ]
        }

    monkeypatch.setattr(search_web_module, "_http_get_json", fake_serp)
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Perplexity fallback should not run in this scenario.")
        ),
    )

    state = create_initial_state(
        user_query=(
            "research ai content workflow trends and create a blog article, "
            "linkedin post, and image concept"
        ),
    )
    result = build_langgraph().invoke(state)

    assert result["final_response"].strip()
    assert result["workflow_status"] in {"success", "partial_success"}
    assert "## Blog Draft" in result["final_response"]
    assert "## LinkedIn Draft" in result["final_response"]
    assert "## Image Assets" in result["final_response"]
    assert calls["serp"] > 0

    urls = [source.get("url") for source in result["sources"] if source.get("url")]
    assert len(urls) == len(set(urls))
    assert "(None)" not in result["final_response"]

    assert result["cost_controls"]["tokens_used_this_session"] > 0
    assert result["cost_controls"]["search_queries_used_this_session"] > 0
    assert any(
        output.get("status") == "success" for output in result.get("image_outputs", [])
    )
    assert (
        result.get("tool_outputs", {}).get("image_agent", {}).get("status") == "success"
    )

    for source in result["sources"]:
        if source.get("citation_available") is False:
            assert source.get("url") in {None, ""}

    for image_output in result.get("image_outputs", []):
        assert "base64" not in image_output
        assert "b64_json" not in image_output
