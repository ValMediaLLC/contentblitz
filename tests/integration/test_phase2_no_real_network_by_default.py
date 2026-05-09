from __future__ import annotations

import importlib
import socket
from types import SimpleNamespace
from urllib import request as urllib_request

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
search_web_module = importlib.import_module("contentblitz.tools.search_web")
perplexity_module = importlib.import_module("contentblitz.tools.perplexity")


def _block_network(*args, **kwargs):
    raise AssertionError("Unexpected real network call attempted during test.")


def _make_text_client():
    def create(**kwargs):
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _make_image_client():
    def generate(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(url="https://img.example/network-safe.png")])

    return SimpleNamespace(images=SimpleNamespace(generate=generate))


def test_no_real_network_calls_when_providers_are_mocked(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", _block_network)
    monkeypatch.setattr(urllib_request, "urlopen", _block_network)
    monkeypatch.setattr(search_web_module.request, "urlopen", _block_network)
    monkeypatch.setattr(perplexity_module.request, "urlopen", _block_network)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")

    monkeypatch.setattr(generate_text_module, "_build_openai_client", lambda api_key: _make_text_client())
    monkeypatch.setattr(generate_image_module, "_build_openai_client", lambda api_key: _make_image_client())
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: {
            "organic_results": [
                {
                    "title": "Network blocked success",
                    "link": "https://example.com/safe",
                    "snippet": "safe snippet",
                    "source": "safe",
                }
            ]
        },
    )

    text_result = generate_text_module.generate_text(
        prompt="network blocked text",
        agent_key="query_handler",
    )
    search_result = search_web_module.search_web("network blocked search", provider="serp")
    image_result = generate_image_module.generate_image(prompt="network blocked image")

    assert text_result.degraded is False
    assert search_result.degraded is False
    assert image_result.degraded is False


def test_missing_keys_short_circuit_without_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", _block_network)
    monkeypatch.setattr(urllib_request, "urlopen", _block_network)
    monkeypatch.setattr(search_web_module.request, "urlopen", _block_network)
    monkeypatch.setattr(perplexity_module.request, "urlopen", _block_network)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    text_result = generate_text_module.generate_text(
        prompt="missing key text",
        agent_key="query_handler",
    )
    serp_result = search_web_module.search_web("missing key serp", provider="serp")
    perplexity_result = search_web_module.search_web("missing key perplexity", provider="perplexity")
    image_result = generate_image_module.generate_image(prompt="missing key image")

    assert text_result.degraded is True
    assert text_result.error["code"] == "configuration_error"
    assert serp_result.degraded is True
    assert serp_result.error["code"] == "configuration_error"
    assert perplexity_result.degraded is True
    assert perplexity_result.error["code"] == "configuration_error"
    assert image_result.degraded is True
    assert image_result.error["code"] == "configuration_error"
