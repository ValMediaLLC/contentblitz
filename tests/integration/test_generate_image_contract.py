from __future__ import annotations

import importlib
from dataclasses import is_dataclass

generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
legacy_image_module = importlib.import_module("contentblitz.tools.image")


class _FakeImageClient:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("No scripted image response remaining.")
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _mock_stability_client(monkeypatch, scripted):
    client = _FakeImageClient(scripted)
    monkeypatch.setattr(
        generate_image_module,
        "_build_stability_client",
        lambda api_key: client,
    )
    return client


def _mock_fal_client(monkeypatch, scripted):
    client = _FakeImageClient(scripted)
    monkeypatch.setattr(
        generate_image_module,
        "_build_fal_client",
        lambda api_key: client,
    )
    return client


def test_generate_image_contract_shape(monkeypatch) -> None:
    monkeypatch.setenv("STABILITY_API_KEY", "stability-test")
    stability_client = _mock_stability_client(
        monkeypatch,
        [
            {
                "images": [
                    {
                        "url": "https://img.example/contract.png",
                        "revised_prompt": "contract revised",
                    }
                ]
            }
        ],
    )

    result = generate_image_module.generate_image(
        prompt="A clean futuristic dashboard concept.",
        model="stable-image-core",
        size="1024x1024",
        quality="standard",
    )

    assert is_dataclass(result)
    assert result.provider == "stability_ai"
    assert result.model == "stable-image-core"
    assert result.prompt == "A clean futuristic dashboard concept."
    assert result.image_url == "https://img.example/contract.png"
    assert result.local_path is None
    assert result.image_id is None
    assert result.renderable is True
    assert result.revised_prompt == "contract revised"
    assert result.degraded is False
    assert result.error is None

    assert len(stability_client.calls) == 1
    call = stability_client.calls[0]
    assert call["model"] == "stable-image-core"
    assert call["prompt"] == "A clean futuristic dashboard concept."
    assert call["size"] == "1024x1024"


def test_legacy_image_adapter_remains_compatible(monkeypatch) -> None:
    monkeypatch.setenv("STABILITY_API_KEY", "stability-test")
    _mock_stability_client(
        monkeypatch,
        [{"images": [{"url": "https://img.example/legacy.png"}]}],
    )

    legacy = legacy_image_module.generate_image(
        prompt="Legacy adapter image prompt.",
        style="editorial",
    )
    assert legacy["provider_primary"] == "stability_ai"
    assert legacy["provider_fallback"] == "fal_ai"
    assert legacy["provider_used"] in {"stability_ai", "fal_ai"}
    assert legacy["used_external_api"] is True
    assert legacy["degraded"] is False
    assert legacy["error"] is None
    assert isinstance(legacy["images"], list)
    assert len(legacy["images"]) == 1
    assert legacy["images"][0]["url"] == "https://img.example/legacy.png"


def test_generate_image_live_calls_disabled_contract(monkeypatch) -> None:
    monkeypatch.setenv("STABILITY_API_KEY", "stability-test")
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "0")

    client_built = {"value": False}

    def _fake_builder(api_key: str):
        _ = api_key
        client_built["value"] = True
        raise AssertionError(
            "Image client should not be built when live calls are disabled."
        )

    monkeypatch.setattr(generate_image_module, "_build_stability_client", _fake_builder)

    result = generate_image_module.generate_image(
        prompt="Disabled-live-call contract case."
    )
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "live_calls_disabled"
    assert client_built["value"] is False


def test_legacy_image_adapter_maps_non_url_image_refs_to_id(monkeypatch) -> None:
    monkeypatch.setenv("STABILITY_API_KEY", "stability-test")
    _mock_stability_client(
        monkeypatch,
        [{"images": [{"id": "asset_123abc"}]}],
    )

    legacy = legacy_image_module.generate_image(
        prompt="Legacy adapter non-url image ref.",
        style="default",
    )
    assert legacy["degraded"] is False
    assert isinstance(legacy["images"], list)
    assert len(legacy["images"]) == 1
    assert legacy["images"][0]["id"] == "asset_123abc"
    assert legacy["images"][0]["renderable"] is False
    assert "url" not in legacy["images"][0]


def test_stability_fallback_to_fal_contract(monkeypatch) -> None:
    monkeypatch.setenv("STABILITY_API_KEY", "stability-test")
    monkeypatch.setenv("FAL_API_KEY", "fal-test")
    stability_client = _mock_stability_client(
        monkeypatch,
        [RuntimeError("stability failed")],
    )
    fal_client = _mock_fal_client(
        monkeypatch,
        [{"images": [{"url": "https://img.example/fal-fallback.png"}]}],
    )

    result = generate_image_module.generate_image(prompt="Fallback contract prompt.")
    assert result.degraded is False
    assert result.provider == "fal_ai"
    assert result.model == "fal-ai/fast-sdxl"
    assert result.image_url == "https://img.example/fal-fallback.png"
    assert len(stability_client.calls) == 1
    assert len(fal_client.calls) == 1
