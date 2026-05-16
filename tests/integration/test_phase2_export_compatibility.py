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
    chunk = (
        "Pipeline quality improves when teams define repeatable steps for research, drafting, "
        "validation, and distribution. "
    )
    return (
        "Your content engine does not need more prompts. It needs better sequencing.\n\n"
        + (chunk * 8)
        + "\n\nWhat stage do you want to standardize first?\n"
        "#AI #ContentMarketing #Ops"
    )


def _text_payload(prompt: str) -> str:
    if "Generate 3-5 search queries as JSON list for this topic" in prompt:
        return json.dumps(
            [
                "ai workflow trends",
                "ai marketing benchmarks",
                "content ops case studies",
            ]
        )
    if "Synthesize a concise research brief from these findings." in prompt:
        return "Research synthesis for export compatibility validation."
    if "Create a JSON content brief for 'blog'" in prompt:
        return json.dumps(
            {
                "format": "blog",
                "objective": "Educate readers",
                "audience": "marketing leaders",
                "tone": "practical",
                "angle": "execution playbook",
            }
        )
    if "Create a JSON content brief for 'linkedin'" in prompt:
        return json.dumps(
            {
                "format": "linkedin",
                "objective": "Spark discussion",
                "audience": "operators",
                "tone": "direct",
                "angle": "operational lessons",
            }
        )
    if "Create a JSON content brief for 'image'" in prompt:
        return json.dumps(
            {
                "format": "image",
                "prompt_focus": "content operations control room",
                "visual_direction": "cinematic realism",
            }
        )
    if "Write an SEO-friendly blog draft in markdown." in prompt:
        return (
            "# Content Workflow Design\n\n"
            "Strong pipelines combine reliable research inputs, explicit content briefs, and structured QA loops."
        )
    if "Write a LinkedIn post in plain text." in prompt:
        return _long_linkedin_post()
    if "Enhance this image generation prompt for clarity and visual detail." in prompt:
        return "Create a high-detail cinematic content operations control room."
    return "Generic mocked output."


def _make_text_client():
    def create(**kwargs):
        content = _text_payload(kwargs["messages"][0]["content"])
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=5, completion_tokens=5, total_tokens=10
            ),
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _make_image_client():
    def generate(**kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(url="https://img.example/export-check.png")]
        )

    return SimpleNamespace(images=SimpleNamespace(generate=generate))


def _export_state_for_outputs(
    outputs: list[str], *, research_required: bool = False
) -> dict:
    return create_initial_state(
        user_query="",
        requested_outputs=outputs,
        research_required=research_required,
        intent="content_creation" if outputs != ["research"] else "research",
        export_requested=True,
        export_metadata={
            "formats_requested": ["markdown", "pdf"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        },
    )


def test_export_still_works_for_phase2_multimodal_output(monkeypatch) -> None:
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
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: {
            "organic_results": [
                {
                    "title": "Export source",
                    "link": "https://example.com/export-source",
                    "snippet": "Detailed source snippet for export compatibility checks.",
                    "source": "Example",
                }
            ]
        },
    )

    result = build_langgraph().invoke(
        _export_state_for_outputs(["blog", "linkedin", "image"], research_required=True)
    )

    assert result["final_response"].strip()
    assert result["workflow_status"] in {"success", "partial_success"}
    assert result["assembled_outputs"]["blog"].strip()
    assert result["assembled_outputs"]["linkedin"].strip()
    assert result["assembled_outputs"]["image"].strip()

    export_outputs = result["export_outputs"]
    assert export_outputs["blog"]["content"].strip()
    assert export_outputs["linkedin"]["content"].strip()
    assert export_outputs["image"]["content"].strip()
    assert export_outputs["blog"]["filename"].endswith(".md")
    assert export_outputs["linkedin"]["filename"].endswith(".md")
    assert export_outputs["image"]["filename"].endswith(".md")

    export_paths = result["export_metadata"]["export_paths"]
    assert export_paths.get("markdown", "").strip()
    assert export_paths.get("pdf", "").strip()

    assert "base64" not in export_outputs["image"]["content"]
    assert "b64_json" not in export_outputs["image"]["content"]


def test_export_still_works_for_research_with_perplexity_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "px-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(),
    )
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
                        "content": "Perplexity fallback produced exportable research summary content.",
                    },
                    "citations": [],
                }
            ]
        },
    )

    state = _export_state_for_outputs(["research"], research_required=True)
    state["user_query"] = "research fallback export path"
    result = build_langgraph().invoke(state)

    assert result["final_response"].strip()
    assert result["workflow_status"] in {"success", "partial_success"}
    assert result["research_data"]["fallback_used"] is True
    assert "(None)" not in result["final_response"]

    research_export = result["export_outputs"]["research"]
    assert research_export["content"].strip()
    assert research_export["filename"].endswith(".md")
    assert "(None)" not in research_export["content"]
