from __future__ import annotations

import importlib
from types import SimpleNamespace

from contentblitz.tools.provider_types import SearchWebResult

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
search_web_module = importlib.import_module("contentblitz.tools.search_web")


def _make_text_client(create_fn):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))


def _make_image_client(generate_fn):
    return SimpleNamespace(images=SimpleNamespace(generate=generate_fn))


def _assert_secret_safe(error_payload: object, *secrets: str) -> None:
    error_text = str(error_payload)
    for secret in secrets:
        assert secret not in error_text
    assert "Traceback" not in error_text
    assert "File \"" not in error_text


def test_generate_text_failure_is_normalized_without_secret_or_stacktrace(monkeypatch):
    secret = "sk-live-super-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def create(**kwargs):
        raise RuntimeError(f"provider exploded with {secret} and internal traceback details")

    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(create),
    )

    result = generate_text_module.generate_text(
        prompt="failure scenario",
        agent_key="query_handler",
    )

    assert result.degraded is True
    assert result.text == ""
    assert result.error is not None
    assert result.error["code"] == "provider_failure"
    assert isinstance(result.error.get("last_error"), dict)
    _assert_secret_safe(result.error, secret)


def test_search_web_failure_is_normalized_without_secret_or_raw_payload(monkeypatch):
    secret = "serp-secret-token-123"
    monkeypatch.setenv("SERP_API_KEY", secret)

    def bad_payload(_url):
        raise RuntimeError(f"raw provider payload leaked {secret}")

    monkeypatch.setattr(search_web_module, "_http_get_json", bad_payload)
    result = search_web_module.search_web("failure query", provider="serp")

    assert isinstance(result, SearchWebResult)
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_error"
    _assert_secret_safe(result.error, secret)


def test_generate_image_failure_is_normalized_without_secret_or_stacktrace(monkeypatch):
    secret = "sk-image-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def generate(**kwargs):
        raise RuntimeError(f"image provider payload included {secret}")

    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: _make_image_client(generate),
    )

    result = generate_image_module.generate_image(prompt="failure image")
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_failure"
    _assert_secret_safe(result.error, secret)


def test_degraded_results_remain_structured_across_tools(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    text_result = generate_text_module.generate_text(
        prompt="missing key text",
        agent_key="query_handler",
    )
    search_result = search_web_module.search_web("missing key web", provider="serp")
    image_result = generate_image_module.generate_image(prompt="missing key image")

    assert text_result.degraded is True
    assert isinstance(text_result.error, dict)
    assert {"code", "message", "provider", "recoverable"} <= set(text_result.error.keys())

    assert search_result.degraded is True
    assert isinstance(search_result.error, dict)
    assert {"code", "message", "provider", "recoverable"} <= set(search_result.error.keys())

    assert image_result.degraded is True
    assert isinstance(image_result.error, dict)
    assert {"code", "message", "provider", "recoverable"} <= set(image_result.error.keys())
