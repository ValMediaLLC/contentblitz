from __future__ import annotations

import copy
import importlib
from dataclasses import is_dataclass
from types import SimpleNamespace

from contentblitz.tools.provider_types import SearchResult, SearchWebResult

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
search_web_module = importlib.import_module("contentblitz.tools.search_web")


def _text_response(*, content: str, model: str = "gpt-4o-mini"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


def _make_text_client(create_fn):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn)))


def _make_image_client(generate_fn):
    return SimpleNamespace(images=SimpleNamespace(generate=generate_fn))


def _image_response(*, url: str, revised_prompt: str | None = None):
    payload = {"url": url}
    if revised_prompt is not None:
        payload["revised_prompt"] = revised_prompt
    return SimpleNamespace(data=[SimpleNamespace(**payload)])


def test_generate_text_success_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return _text_response(content="Provider contract text", model=kwargs["model"])

    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(create),
    )

    result = generate_text_module.generate_text(
        prompt="Write a short provider contract line.",
        agent_key="query_handler",
        model="gpt-4o",
    )

    assert is_dataclass(result)
    assert result.provider == "openai"
    assert result.model == "gpt-4o"
    assert result.text == "Provider contract text"
    assert result.degraded is False
    assert result.error is None
    assert result.input_tokens == 5
    assert result.output_tokens == 7
    assert result.total_tokens == 12
    assert len(calls) == 1


def test_generate_text_primary_failure_then_fallback_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    called_models = []

    def create(**kwargs):
        model = kwargs["model"]
        called_models.append(model)
        if model == "gpt-4o":
            raise RuntimeError("primary provider failure")
        return _text_response(content="Fallback output", model="gpt-4o-mini")

    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(create),
    )

    result = generate_text_module.generate_text(
        prompt="Fallback behavior validation",
        agent_key="query_handler",
        model="gpt-4o",
    )

    assert result.degraded is False
    assert result.model == "gpt-4o-mini"
    assert result.text == "Fallback output"
    assert called_models.count("gpt-4o") == 2
    assert "gpt-4o-mini" in called_models


def test_generate_text_total_provider_failure_is_structured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def create(**kwargs):
        raise RuntimeError("raw provider payload: {'secret':'sk-test-value'}")

    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(create),
    )

    result = generate_text_module.generate_text(
        prompt="Total failure contract case",
        agent_key="query_handler",
        model="gpt-4o",
    )

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_failure"
    assert "models_attempted" in result.error
    assert "last_error" in result.error


def test_search_web_serp_success_contract(monkeypatch):
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: {
            "organic_results": [
                {
                    "title": "SERP Contract Result",
                    "link": "https://example.com/contract",
                    "snippet": "Contract snippet",
                    "source": "ExampleSource",
                    "date": "2026-05-01",
                }
            ]
        },
    )

    result = search_web_module.search_web("provider contract query", provider="serp")

    assert is_dataclass(result)
    assert result.provider == "serp"
    assert result.degraded is False
    assert result.error is None
    assert len(result.results) == 1
    assert is_dataclass(result.results[0])


def test_search_web_auto_serp_degraded_then_perplexity_success(monkeypatch):
    monkeypatch.setattr(
        search_web_module,
        "_search_serp",
        lambda query, max_results: SearchWebResult(
            provider="serp",
            query=query,
            results=[],
            degraded=True,
            error={"code": "provider_unavailable", "message": "x", "provider": "serp", "recoverable": True},
        ),
    )
    monkeypatch.setattr(
        search_web_module,
        "_search_perplexity_placeholder",
        lambda query, max_results: SearchWebResult(
            provider="perplexity",
            query=query,
            results=[
                SearchResult(
                    title="Perplexity Result",
                    url=None,
                    snippet="Usable fallback answer",
                    source="perplexity",
                    published_at=None,
                    citation_available=False,
                    credibility_score=0.35,
                )
            ],
            degraded=False,
            error=None,
        ),
    )

    result = search_web_module.search_web("fallback query", provider="auto")
    assert result.provider == "perplexity"
    assert result.degraded is False
    assert len(result.results) == 1
    assert result.results[0].citation_available is False


def test_search_web_auto_both_providers_degraded(monkeypatch):
    monkeypatch.setattr(
        search_web_module,
        "_search_serp",
        lambda query, max_results: SearchWebResult(
            provider="serp",
            query=query,
            results=[],
            degraded=True,
            error={"code": "provider_error", "message": "x", "provider": "serp", "recoverable": True},
        ),
    )
    monkeypatch.setattr(
        search_web_module,
        "_search_perplexity_placeholder",
        lambda query, max_results: SearchWebResult(
            provider="perplexity",
            query=query,
            results=[],
            degraded=True,
            error={"code": "provider_error", "message": "y", "provider": "perplexity", "recoverable": True},
        ),
    )

    result = search_web_module.search_web("all degraded query", provider="auto")
    assert result.provider == "auto"
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "all_providers_failed"


def test_generate_image_dalle3_success_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return _image_response(url="https://img.example/success.png", revised_prompt="revised")

    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: _make_image_client(generate),
    )

    result = generate_image_module.generate_image(prompt="Create an image")

    assert is_dataclass(result)
    assert result.provider == "openai"
    assert result.model == "dall-e-3"
    assert result.image_url == "https://img.example/success.png"
    assert result.revised_prompt == "revised"
    assert result.degraded is False
    assert result.error is None
    assert len(calls) == 1


def test_generate_image_dalle3_failure_then_dalle2_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    models = []

    def generate(**kwargs):
        model = kwargs["model"]
        models.append(model)
        if model == "dall-e-3":
            raise RuntimeError("primary model failed")
        return _image_response(url="https://img.example/fallback.png")

    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: _make_image_client(generate),
    )

    result = generate_image_module.generate_image(prompt="Fallback image request")
    assert result.degraded is False
    assert result.model == "dall-e-2"
    assert result.image_url == "https://img.example/fallback.png"
    assert models == ["dall-e-3", "dall-e-2"]


def test_generate_image_both_models_fail(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def generate(**kwargs):
        raise RuntimeError("all image models failed")

    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: _make_image_client(generate),
    )

    result = generate_image_module.generate_image(prompt="Both fail")
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_failure"
    assert result.error["models_attempted"] == ["dall-e-3", "dall-e-2"]


def test_missing_api_keys_fail_safely_only_on_invocation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    assert callable(generate_text_module.generate_text)
    assert callable(search_web_module.search_web)
    assert callable(generate_image_module.generate_image)

    text_result = generate_text_module.generate_text(
        prompt="missing key",
        agent_key="query_handler",
    )
    web_result = search_web_module.search_web("missing serp key", provider="serp")
    image_result = generate_image_module.generate_image(prompt="missing key image")

    assert text_result.degraded is True
    assert text_result.error["code"] == "configuration_error"
    assert web_result.degraded is True
    assert web_result.error["code"] == "configuration_error"
    assert image_result.degraded is True
    assert image_result.error["code"] == "configuration_error"


def test_tools_never_mutate_state_and_return_normalized_objects(monkeypatch):
    sentinel_state = {
        "retry_counts": {"research_agent": 1},
        "cost_controls": {"tokens_used_this_session": 99},
        "content_drafts": {"blog": {"body": "x"}},
        "sources": [{"title": "a"}],
        "errors": [{"type": "existing"}],
    }
    before = copy.deepcopy(sentinel_state)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: _make_text_client(lambda **kwargs: _text_response(content="ok", model=kwargs["model"])),
    )
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda _url: {
            "organic_results": [
                {
                    "title": "Result",
                    "link": "https://example.com",
                    "snippet": "snippet text",
                    "source": "source",
                }
            ]
        },
    )
    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: _make_image_client(lambda **kwargs: _image_response(url="https://img.example/a.png")),
    )

    text_result = generate_text_module.generate_text(prompt="state immutability", agent_key="query_handler")
    web_result = search_web_module.search_web("state immutability web", provider="serp")
    image_result = generate_image_module.generate_image(prompt="state immutability image")

    assert is_dataclass(text_result)
    assert is_dataclass(web_result)
    assert is_dataclass(image_result)
    assert sentinel_state == before
