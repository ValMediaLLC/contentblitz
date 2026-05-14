from __future__ import annotations

import importlib
import json
import socket
from types import SimpleNamespace
from urllib import request as urllib_request

from frontend.services.orchestrator_client import stream_workflow_progress

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
search_web_module = importlib.import_module("contentblitz.tools.search_web")
perplexity_module = importlib.import_module("contentblitz.tools.perplexity")
orchestrator_client_module = importlib.import_module("frontend.services.orchestrator_client")


def block_network(*args, **kwargs):
    raise AssertionError("Unexpected real network call attempted.")


def apply_no_network(monkeypatch) -> None:
    monkeypatch.setattr(socket, "create_connection", block_network)
    monkeypatch.setattr(urllib_request, "urlopen", block_network)
    monkeypatch.setattr(search_web_module.request, "urlopen", block_network)
    monkeypatch.setattr(perplexity_module.request, "urlopen", block_network)


def reset_orchestrator_graph() -> None:
    orchestrator_client_module._GRAPH = None


def _long_linkedin_post() -> str:
    section = (
        "High-performing content teams align research, drafting, QA, and distribution "
        "inside one repeatable operating rhythm. "
    )
    return (
        "AI content workflows break when ownership is unclear.\n\n"
        + (section * 8)
        + "\n\nWhere does your workflow lose momentum right now?\n"
        "#AI #ContentOps #MarketingStrategy"
    )


def default_text_payload(prompt: str) -> str:
    lowered = prompt.lower()
    if "classify the user request for contentblitz." in lowered:
        if "futuristic apparel image concepts" in lowered:
            outputs = ["blog", "linkedin", "research", "image"]
        elif "research ai content marketing trends for 2026" in lowered:
            outputs = ["research"]
        elif "blog article about ai productivity tools" in lowered:
            outputs = ["blog", "research"]
        else:
            outputs = ["blog", "linkedin"]
        return json.dumps(
            {
                "intent": "content_creation",
                "requested_outputs": outputs,
                "research_required": "research" in outputs or outputs != ["image"],
                "clarification_needed": False,
                "clarification_message": None,
                "export_requested": False,
            }
        )
    if "generate 3-5 search queries as json list for this topic" in lowered:
        return json.dumps(
            [
                "ai productivity workflow benchmarks",
                "ai content systems trends 2026",
                "content operations case studies",
            ]
        )
    if "synthesize a concise research brief from these findings." in lowered:
        return "Research synthesis from mocked sources with deterministic confidence notes."
    if "create a json content brief for 'blog'" in lowered:
        return json.dumps(
            {
                "format": "blog",
                "objective": "Educate operators with practical implementation detail.",
                "audience": "marketing operators",
                "tone": "clear and practical",
                "angle": "repeatable systems design",
            }
        )
    if "create a json content brief for 'linkedin'" in lowered:
        return json.dumps(
            {
                "format": "linkedin",
                "objective": "Drive discussion among operations leaders.",
                "audience": "marketing leaders",
                "tone": "direct",
                "angle": "operational playbook",
            }
        )
    if "create a json content brief for 'image'" in lowered:
        return json.dumps(
            {
                "format": "image",
                "prompt_focus": "futuristic apparel campaign concept art",
                "visual_direction": "cinematic, high-contrast studio aesthetic",
            }
        )
    if "write an seo-friendly blog draft in markdown." in lowered:
        return (
            "# AI Productivity Workflows\n\n"
            "Repeatable editorial workflows outperform ad-hoc prompting because ownership, "
            "quality gates, and feedback loops are explicit."
        )
    if "write a linkedin post in plain text." in lowered:
        return _long_linkedin_post()
    if "enhance this image generation prompt for clarity and visual detail." in lowered:
        return "Create a high-detail futuristic apparel campaign scene with cinematic contrast."
    return "Generic mocked provider output."


def install_mock_text_client(monkeypatch, *, total_tokens: int = 12) -> None:
    def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        payload = default_text_payload(prompt)
        prompt_tokens = max(1, total_tokens // 2)
        completion_tokens = max(0, total_tokens - prompt_tokens)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(generate_text_module, "_build_openai_client", lambda api_key: client)


def install_mock_search(monkeypatch, *, weak_serp: bool = False) -> dict[str, int]:
    counters = {"serp": 0, "perplexity": 0}

    def fake_serp(_url: str):
        counters["serp"] += 1
        if weak_serp:
            return {
                "organic_results": [
                    {
                        "title": "Weak source",
                        "link": "https://example.com/weak-source",
                        "snippet": "short",
                        "source": "Example",
                    }
                ]
            }
        return {
            "organic_results": [
                {
                    "title": "Primary source A",
                    "link": "https://example.com/source-a",
                    "snippet": "Detailed source snippet about AI workflow execution and measurable gains.",
                    "source": "ExampleA",
                    "date": "2026-05-10",
                },
                {
                    "title": "Primary source A duplicate",
                    "link": "https://example.com/source-a",
                    "snippet": "Duplicate URL entry should be deduped downstream.",
                    "source": "ExampleA",
                },
                {
                    "title": "Primary source B",
                    "link": "https://example.com/source-b",
                    "snippet": "Additional detailed source text for citation validation and synthesis.",
                    "source": "ExampleB",
                },
            ]
        }

    def fake_perplexity(**_kwargs):
        counters["perplexity"] += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": "Fallback source synthesis for degraded provider conditions.",
                    },
                    "citations": [],
                }
            ]
        }

    monkeypatch.setattr(search_web_module, "_http_get_json", fake_serp)
    monkeypatch.setattr(perplexity_module, "_http_post_json", fake_perplexity)
    return counters


def install_mock_image_client(monkeypatch, *, fail_all: bool = False) -> dict[str, list[str]]:
    calls = {"models": []}

    def generate(**kwargs):
        model = kwargs["model"]
        calls["models"].append(model)
        if fail_all:
            raise RuntimeError("forced image provider failure")
        return SimpleNamespace(
            data=[SimpleNamespace(url="https://img.example/contentblitz-ui-integration.png")]
        )

    client = SimpleNamespace(images=SimpleNamespace(generate=generate))
    monkeypatch.setattr(generate_image_module, "_build_openai_client", lambda api_key: client)
    return calls


def collect_stream_result(
    *,
    user_query: str,
    requested_outputs: list[str],
    export_requested: bool = False,
    export_formats: list[str] | None = None,
) -> tuple[list[dict], dict]:
    events: list[dict] = []
    final_result: dict = {}
    for item in stream_workflow_progress(
        user_query=user_query,
        requested_outputs=requested_outputs,
        export_requested=export_requested,
        export_formats=export_formats or [],
    ):
        if item.get("type") == "progress":
            event = item.get("event")
            if isinstance(event, dict):
                events.append(event)
        elif item.get("type") == "final":
            payload = item.get("result")
            if isinstance(payload, dict):
                final_result = payload
            raw_events = item.get("events")
            if isinstance(raw_events, list):
                events = [entry for entry in raw_events if isinstance(entry, dict)]
    return events, final_result
