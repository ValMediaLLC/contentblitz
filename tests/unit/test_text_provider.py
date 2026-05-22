from __future__ import annotations

import importlib
from types import SimpleNamespace

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")


def _build_anthropic_response(
    *,
    text: str,
    model: str = "claude-3-5-sonnet-latest",
    input_tokens: int | None = 9,
    output_tokens: int | None = 11,
    cache_creation_input_tokens: int | None = 3,
    cache_read_input_tokens: int | None = 4,
):
    usage = None
    if input_tokens is not None and output_tokens is not None:
        usage = SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
    return SimpleNamespace(
        model=model,
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage,
    )


def _build_openai_response(
    *,
    text: str,
    model: str = "gpt-4o-mini",
):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


def test_anthropic_success_response_is_normalized(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv("CONTENTBLITZ_TEXT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    calls: list[dict[str, object]] = []

    def _create(**kwargs):
        calls.append(kwargs)
        return _build_anthropic_response(text="Anthropic output", model=kwargs["model"])

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=_create))
    monkeypatch.setattr(
        generate_text_module,
        "_build_anthropic_client",
        lambda api_key: fake_client,
    )

    result = generate_text_module.generate_text(
        prompt="Write with Anthropic provider.",
        agent_key="blog_writer",
    )

    assert result.degraded is False
    assert result.provider == "anthropic"
    assert result.text == "Anthropic output"
    assert result.total_tokens == 20
    assert result.cache_creation_input_tokens == 3
    assert result.cache_read_input_tokens == 4
    assert calls and calls[0]["model"] == "claude-3-5-sonnet-latest"


def test_anthropic_missing_usage_fields_safe_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv("CONTENTBLITZ_TEXT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: _build_anthropic_response(
                text="No usage",
                model=kwargs["model"],
                input_tokens=None,
                output_tokens=None,
            )
        )
    )
    monkeypatch.setattr(
        generate_text_module,
        "_build_anthropic_client",
        lambda api_key: fake_client,
    )

    result = generate_text_module.generate_text(
        prompt="No usage payload.",
        agent_key="query_handler",
    )

    assert result.degraded is False
    assert result.provider == "anthropic"
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.total_tokens == 0
    assert result.cache_creation_input_tokens == 0
    assert result.cache_read_input_tokens == 0


def test_anthropic_provider_error_is_safe_and_redacted(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv(
        "CONTENTBLITZ_AGENT_MODEL_POLICY",
        (
            '{"query_handler":{"provider":"anthropic","model":"claude-3-5-sonnet-latest",'
            '"fallback_provider":"anthropic","fallback_model":"claude-3-5-haiku-latest"}}'
        ),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    def _create(**kwargs):
        raise RuntimeError("provider payload secret sk-live-123 should not leak")

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=_create))
    monkeypatch.setattr(
        generate_text_module,
        "_build_anthropic_client",
        lambda api_key: fake_client,
    )

    result = generate_text_module.generate_text(
        prompt="Force provider failure.",
        agent_key="query_handler",
    )

    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "unknown_provider_error"
    assert "sk-live-123" not in str(result.error)
    assert "traceback" not in str(result.error).lower()


def test_generate_text_routes_to_anthropic_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv("CONTENTBLITZ_TEXT_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")

    openai_called = {"value": False}
    anthropic_called = {"value": False}

    def _openai_builder(api_key):
        openai_called["value"] = True
        raise AssertionError("OpenAI builder should not be used for Anthropic routing.")

    def _anthropic_builder(api_key):
        anthropic_called["value"] = True
        return SimpleNamespace(
            messages=SimpleNamespace(
                create=lambda **kwargs: _build_anthropic_response(
                    text="Anthropic route", model=kwargs["model"]
                )
            )
        )

    monkeypatch.setattr(generate_text_module, "_build_openai_client", _openai_builder)
    monkeypatch.setattr(
        generate_text_module,
        "_build_anthropic_client",
        _anthropic_builder,
    )

    result = generate_text_module.generate_text(
        prompt="Provider routing validation.",
        agent_key="content_strategist",
    )

    assert result.degraded is False
    assert result.provider == "anthropic"
    assert anthropic_called["value"] is True
    assert openai_called["value"] is False


def test_missing_anthropic_key_falls_back_to_openai_when_configured(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv("CONTENTBLITZ_TEXT_PROVIDER", "anthropic")
    monkeypatch.setenv(
        "CONTENTBLITZ_AGENT_MODEL_POLICY",
        (
            '{"blog_writer":{"provider":"anthropic","model":"claude-sonnet-4-6",'
            '"fallback_provider":"openai","fallback_model":"gpt-4o-mini"}}'
        ),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    openai_calls: list[dict[str, object]] = []

    def _openai_create(**kwargs):
        openai_calls.append(kwargs)
        return _build_openai_response(text="OpenAI fallback", model=kwargs["model"])

    fake_openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_openai_create))
    )
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: fake_openai_client,
    )
    monkeypatch.setattr(
        generate_text_module,
        "_build_anthropic_client",
        lambda api_key: (_ for _ in ()).throw(
            AssertionError("Anthropic builder should not be called without key.")
        ),
    )

    result = generate_text_module.generate_text(
        prompt="Fallback to OpenAI when Anthropic key is missing.",
        agent_key="blog_writer",
    )

    assert result.degraded is False
    assert result.provider == "openai"
    assert result.text == "OpenAI fallback"
    assert openai_calls


def test_missing_anthropic_key_returns_safe_configuration_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv("CONTENTBLITZ_TEXT_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = generate_text_module.generate_text(
        prompt="Anthropic key missing diagnostic test.",
        agent_key="blog_writer",
    )

    assert result.degraded is True
    assert result.provider == "anthropic"
    assert result.error is not None
    assert result.error["code"] == "configuration_error"
    assert result.error["provider"] == "anthropic"
    assert result.error["requested_provider"] == "anthropic"
    assert result.error["fallback_provider"] == "anthropic"
    assert "traceback" not in str(result.error).lower()
    assert "api_key=" not in str(result.error).lower()
