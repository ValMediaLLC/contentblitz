from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

from contentblitz.tools import perplexity as perplexity_module


def test_missing_api_key_fails_safely_on_invocation(monkeypatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    result = perplexity_module.search_perplexity("ai workflows", max_results=3)

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "configuration_error"


def test_live_calls_disabled_fails_safely_without_network(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "0")

    def _fail_if_called(**_kwargs):
        raise AssertionError("HTTP layer should not be called when live calls are disabled.")

    monkeypatch.setattr(perplexity_module, "_http_post_json", _fail_if_called)
    result = perplexity_module.search_perplexity("ai workflows", max_results=3)

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "live_calls_disabled"


def test_malformed_provider_payload_returns_degraded(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(
        perplexity_module, "_http_post_json", lambda **_kwargs: {"choices": []}
    )

    result = perplexity_module.search_perplexity("ai workflows", max_results=3)

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_payload_unusable"


def test_empty_answer_returns_degraded(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **_kwargs: {
            "choices": [{"message": {"content": "   "}}],
            "citations": [],
        },
    )

    result = perplexity_module.search_perplexity("ai workflows", max_results=3)

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_payload_unusable"


def test_missing_citations_do_not_create_fake_urls(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **_kwargs: {
            "choices": [{"message": {"content": "Safe fallback answer"}}]
        },
    )

    result = perplexity_module.search_perplexity("ai workflows", max_results=3)

    assert result.degraded is False
    assert len(result.results) == 1
    first = result.results[0]
    assert first.url is None
    assert first.citation_available is False


def test_provider_http_error_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")

    def _raise_http_error(**_kwargs):
        raise HTTPError(
            url="https://api.perplexity.ai/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(perplexity_module, "_http_post_json", _raise_http_error)
    result = perplexity_module.search_perplexity("ai workflows", max_results=3)

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_http_error"
    assert result.error["status_code"] == 429


def test_provider_url_error_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **_kwargs: (_ for _ in ()).throw(URLError("network down")),
    )

    result = perplexity_module.search_perplexity("ai workflows", max_results=3)
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_unavailable"


def test_provider_json_decode_error_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **_kwargs: (_ for _ in ()).throw(json.JSONDecodeError("bad", "{}", 0)),
    )

    result = perplexity_module.search_perplexity("ai workflows", max_results=3)
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_payload_invalid"


def test_provider_exception_is_normalized_without_payload_or_secret_leak(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-secret-value")

    def _raise_generic(**_kwargs):
        raise RuntimeError("raw payload OPENAI_API_KEY=abc123")

    monkeypatch.setattr(perplexity_module, "_http_post_json", _raise_generic)
    result = perplexity_module.search_perplexity("ai workflows", max_results=3)

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_error"
    assert "OPENAI_API_KEY" not in str(result.error)
    assert "abc123" not in str(result.error)
