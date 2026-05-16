from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping

from frontend.components import result_view as result_view_module


@dataclass
class _DummyStreamlit:
    markdown_calls: List[str] = field(default_factory=list)
    image_calls: List[tuple[str, str]] = field(default_factory=list)

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def markdown(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.markdown_calls.append(str(value))

    def image(self, image: str, *, caption: str = "", **_kwargs: Any) -> None:
        self.image_calls.append((str(image), str(caption)))


def _render_payload_with_images(
    image_outputs: List[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "partial_output_mode": "none",
        "partial_output_sections": [],
        "image_prompts": [],
        "image_outputs": image_outputs,
    }


def test_image_output_with_url_renders_streamlit_image(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = _render_payload_with_images(
        [
            {
                "status": "success",
                "provider": "dall-e-3",
                "url": "https://img.example/a.png",
                "renderable": True,
            }
        ]
    )
    result_view_module.render_partial_outputs(payload)

    assert dummy_st.image_calls == [
        ("https://img.example/a.png", "dall-e-3 (completed)")
    ]


def test_image_output_with_local_path_renders_streamlit_image(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = _render_payload_with_images(
        [
            {
                "status": "success",
                "provider": "gpt-image-1",
                "local_path": "exports/images/example.png",
                "renderable": True,
            }
        ]
    )
    result_view_module.render_partial_outputs(payload)

    assert dummy_st.image_calls == [
        ("exports/images/example.png", "gpt-image-1 (completed)")
    ]


def test_asset_id_only_does_not_render_streamlit_image(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = _render_payload_with_images(
        [
            {
                "status": "degraded",
                "provider": "gpt-image-1",
                "id": "img_asset_only_001",
                "renderable": False,
            }
        ]
    )
    result_view_module.render_partial_outputs(payload)

    assert dummy_st.image_calls == []
    assert any(
        "non-renderable asset reference" in line for line in dummy_st.markdown_calls
    )
