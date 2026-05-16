from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping

from frontend.components import result_view as result_view_module


@dataclass
class _DummyStreamlit:
    markdown_calls: List[str] = field(default_factory=list)
    image_calls: List[tuple[str, str]] = field(default_factory=list)
    info_calls: List[str] = field(default_factory=list)
    success_calls: List[str] = field(default_factory=list)

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def markdown(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.markdown_calls.append(str(value))

    def image(self, image: str, *, caption: str = "", **_kwargs: Any) -> None:
        self.image_calls.append((str(image), str(caption)))

    def caption(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.info_calls.append(str(value))

    def success(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.success_calls.append(str(value))

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _render_payload_with_images(
    image_outputs: List[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "partial_output_mode": "none",
        "partial_output_sections": [],
        "image_prompts": [],
        "image_outputs": image_outputs,
    }


def _load_session_fixture() -> dict[str, Any]:
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "session_success_empty_partial_outputs.json"
    )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


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


def test_blog_output_is_rendered_once_and_prefers_final_assembled_content(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "final_response": (
            "## Blog Draft\nFinal assembled blog body.\n\n## Sources\n1. x"
        ),
        "partial_output_sections": [
            {
                "key": "blog",
                "label": "Blog Draft",
                "content": "Draft fallback blog body.",
            }
        ],
        "partial_output_mode": "blog_only",
        "image_prompts": [],
        "image_outputs": [],
    }

    result_view_module.render_partial_outputs(payload)
    result_view_module.render_final_response(payload)

    assert (
        sum("Final assembled blog body." in call for call in dummy_st.markdown_calls)
        == 1
    )
    assert not any(
        "Draft fallback blog body." in call for call in dummy_st.markdown_calls
    )


def test_sources_are_rendered_once_from_deduped_payload(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "final_response": (
            "## Blog Draft\nBody text.\n\n## Sources\n"
            "1. [Final Response Source](https://example.com/final)"
        ),
        "sources": [
            {
                "title": "Display Source",
                "url": "https://example.com/display",
                "snippet": "display snippet",
                "citation_available": True,
            }
        ],
    }

    result_view_module.render_final_response(payload)
    result_view_module.render_sources(payload)

    assert sum("#### Sources" in call for call in dummy_st.markdown_calls) == 1
    assert any("Display Source" in call for call in dummy_st.markdown_calls)
    assert not any("Final Response Source" in call for call in dummy_st.markdown_calls)


def test_completed_session_with_empty_partial_outputs_still_renders_final_response(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = _load_session_fixture()
    result_view_module.render_partial_outputs(payload)
    result_view_module.render_final_response(payload)

    assert any(
        "Harnessing the Power of LangSmith for LLM Observability" in call
        for call in dummy_st.markdown_calls
    )
    assert not any(
        "No final response is currently available." in call
        for call in dummy_st.info_calls
    )


def test_result_render_helpers_do_not_mutate_payload(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "partial_success",
        "final_response": "## Blog Draft\nSafe content.",
        "partial_output_mode": "blog_only",
        "partial_output_sections": [
            {"key": "blog", "label": "Blog Draft", "content": "Draft content."}
        ],
        "image_prompts": ["Prompt A"],
        "image_outputs": [
            {
                "status": "success",
                "provider": "dall-e-3",
                "url": "https://img.example/a.png",
                "renderable": True,
            }
        ],
        "sources": [
            {
                "title": "Source A",
                "url": "https://example.com/a",
                "snippet": "snippet",
                "citation_available": True,
            }
        ],
        "export_status": {
            "requested": True,
            "paths": {"markdown": "exports/content.md"},
            "errors": [],
            "non_blocking_failure": False,
        },
    }
    before = copy.deepcopy(payload)

    result_view_module.render_partial_outputs(payload)
    result_view_module.render_final_response(payload)
    result_view_module.render_sources(payload)
    result_view_module.render_export_status(payload)

    assert payload == before


def test_execution_indicators_render_compact_cards_with_truncated_routing(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_execution_indicators(
        execution_status="running",
        result={
            "ui_workflow_status": "success",
            "routing_decision": (
                "research_agent_node_to_content_strategist_node_with_extra_detail"
            ),
        },
    )

    assert any("cbx-metric-grid" in call for call in dummy_st.markdown_calls)
    assert any("Execution" in call for call in dummy_st.markdown_calls)
    assert any("Workflow Status" in call for call in dummy_st.markdown_calls)
    assert any("Routing" in call for call in dummy_st.markdown_calls)
    assert any(
        "research_agent_node_to_conten..." in call
        for call in dummy_st.markdown_calls
    )


def test_usage_summary_renders_compact_cards(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_usage_summary(
        {
            "usage_summary": {
                "estimated_tokens_in": 1200,
                "estimated_tokens_out": 3400,
                "search_queries": 5,
                "image_generation_requests": 2,
                "degraded_operations": 1,
                "retry_attempts": 0,
                "image_generation_failures": 0,
                "sources_returned": 8,
                "export_generation_count": 1,
                "budget_state": "normal",
                "estimated_workflow_cost_level": "medium",
            }
        }
    )

    assert any("cbx-metric-grid" in call for call in dummy_st.markdown_calls)
    assert any("Estimated Tokens" in call for call in dummy_st.markdown_calls)
    assert any("~4,600" in call for call in dummy_st.markdown_calls)
    assert any("Estimated Cost Level" in call for call in dummy_st.markdown_calls)


def test_truncate_display_value_caps_text_safely() -> None:
    assert (
        result_view_module._truncate_display_value(
            "abcdefghijklmnopqrstuvwxyz",
            max_length=12,
        )
        == "abcdefghi..."
    )
