from __future__ import annotations

import importlib
from dataclasses import is_dataclass
from types import SimpleNamespace

generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
legacy_image_module = importlib.import_module("contentblitz.tools.image")


class _FakeImages:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _mock_success_client(monkeypatch, *, url: str, revised_prompt: str | None = None):
    item = {"url": url}
    if revised_prompt is not None:
        item["revised_prompt"] = revised_prompt
    response = SimpleNamespace(data=[SimpleNamespace(**item)])
    images = _FakeImages(response)
    client = SimpleNamespace(images=images)
    monkeypatch.setattr(generate_image_module, "_build_openai_client", lambda api_key: client)
    return images


def test_generate_image_contract_shape(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    images = _mock_success_client(
        monkeypatch,
        url="https://img.example/contract.png",
        revised_prompt="contract revised",
    )

    result = generate_image_module.generate_image(
        prompt="A clean futuristic dashboard concept.",
        model="dall-e-3",
        size="1024x1024",
        quality="standard",
    )

    assert is_dataclass(result)
    assert result.provider == "openai"
    assert result.model == "dall-e-3"
    assert result.prompt == "A clean futuristic dashboard concept."
    assert result.image_url == "https://img.example/contract.png"
    assert result.revised_prompt == "contract revised"
    assert result.degraded is False
    assert result.error is None

    assert len(images.calls) == 1
    call = images.calls[0]
    assert call["model"] == "dall-e-3"
    assert call["prompt"] == "A clean futuristic dashboard concept."
    assert call["size"] == "1024x1024"
    assert call["response_format"] == "url"


def test_legacy_image_adapter_remains_compatible(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _mock_success_client(
        monkeypatch,
        url="https://img.example/legacy.png",
    )

    legacy = legacy_image_module.generate_image(
        prompt="Legacy adapter image prompt.",
        style="editorial",
    )
    assert legacy["provider_primary"] == "dall-e-3"
    assert legacy["provider_fallback"] == "dall-e-2"
    assert legacy["provider_used"] in {"dall-e-3", "dall-e-2"}
    assert legacy["used_external_api"] is True
    assert legacy["degraded"] is False
    assert legacy["error"] is None
    assert isinstance(legacy["images"], list)
    assert len(legacy["images"]) == 1
    assert legacy["images"][0]["url"] == "https://img.example/legacy.png"
