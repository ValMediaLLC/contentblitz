from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping

from frontend.components import result_view as result_view_module


@dataclass
class _DummyContextManager:
    def __enter__(self) -> "_DummyContextManager":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


@dataclass
class _DummyStreamlit:
    markdown_calls: List[str] = field(default_factory=list)
    image_calls: List[tuple[str, str]] = field(default_factory=list)
    info_calls: List[str] = field(default_factory=list)
    success_calls: List[str] = field(default_factory=list)
    warning_calls: List[str] = field(default_factory=list)
    error_calls: List[str] = field(default_factory=list)
    expander_calls: List[dict[str, Any]] = field(default_factory=list)
    json_calls: List[Any] = field(default_factory=list)

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

    def warning(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.warning_calls.append(str(value))

    def error(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.error_calls.append(str(value))

    def expander(
        self, label: str, *, expanded: bool = False, **_kwargs: Any
    ) -> _DummyContextManager:
        self.expander_calls.append({"label": str(label), "expanded": bool(expanded)})
        return _DummyContextManager()

    def write(self, value: Any, *_args: Any, **_kwargs: Any) -> None:
        self.markdown_calls.append(str(value))

    def json(self, value: Any, *_args: Any, **_kwargs: Any) -> None:
        self.json_calls.append(value)


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


def test_base64_image_payload_is_not_rendered(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = _render_payload_with_images(
        [
            {
                "status": "success",
                "provider": "gpt-image-1",
                "url": "data:image/png;base64,AAAA",
                "renderable": True,
            }
        ]
    )
    result_view_module.render_partial_outputs(payload)

    assert dummy_st.image_calls == []
    assert any("hidden for safety" in call.lower() for call in dummy_st.warning_calls)


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


def test_workflow_section_renders_messages_usage_and_result(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "success",
        "final_response": "## Blog Draft\nBody",
        "usage_summary": {
            "estimated_tokens_in": 1000,
            "estimated_tokens_out": 2000,
            "search_queries": 2,
            "image_generation_requests": 0,
            "degraded_operations": 0,
            "retry_attempts": 0,
            "image_generation_failures": 0,
            "sources_returned": 1,
            "export_generation_count": 0,
            "budget_state": "normal",
            "estimated_workflow_cost_level": "low",
        },
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=["Workflow completed successfully."],
        execution_status="success",
        indicator_result={
            "ui_workflow_status": "success",
            "routing_decision": "n/a",
        },
        node_statuses={"query_handler_node": "completed"},
        progress_events=[],
        raw_state={"requested_outputs": ["blog"]},
        raw_submission={"requested_outputs": ["blog"]},
    )

    assert any(
        call["label"] == "Workflow" and call["expanded"]
        for call in dummy_st.expander_calls
    )
    assert any("Status: `success`" in call for call in dummy_st.markdown_calls)
    assert any("Estimated Tokens" in call for call in dummy_st.markdown_calls)
    assert any(
        "Workflow completed successfully." in call for call in dummy_st.info_calls
    )


def test_sectioned_blog_output_renders_once_and_prefers_final_response(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "success",
        "final_response": "## Blog Draft\nFinal blog body.",
        "partial_output_sections": [
            {
                "key": "blog",
                "label": "Blog Draft",
                "content": "Fallback draft body.",
            }
        ],
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=[],
        execution_status="success",
        indicator_result={},
        node_statuses={},
        progress_events=[],
        raw_state={"requested_outputs": ["blog"]},
        raw_submission={},
    )

    assert sum("Final blog body." in call for call in dummy_st.markdown_calls) == 1
    assert not any("Fallback draft body." in call for call in dummy_st.markdown_calls)


def test_sectioned_sources_render_once_with_source_cards(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "success",
        "final_response": "",
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [
            {
                "title": "Source A",
                "url": "https://example.com/a",
                "snippet": "Snippet A",
                "source": "example",
            }
        ],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=[],
        execution_status="success",
        indicator_result={},
        node_statuses={},
        progress_events=[],
        raw_state={},
        raw_submission={},
    )

    assert (
        sum(call["label"] == "Sources" for call in dummy_st.expander_calls)
        == 1
    )
    assert any("cbx-source-card" in call for call in dummy_st.markdown_calls)


def test_images_section_renders_prompts_outputs_and_recoverable_warning(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "partial_success",
        "final_response": "",
        "usage_summary": {},
        "image_prompts": ["Create product hero image"],
        "image_outputs": [
            {
                "status": "success",
                "provider": "dall-e-3",
                "url": "https://img.example/a.png",
                "renderable": True,
            },
            {
                "status": "failed",
                "provider": "dall-e-3",
                "error": {
                    "message": "Image generation encountered a recoverable issue.",
                    "recoverable": True,
                },
            },
        ],
        "sources": [],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=[],
        execution_status="partial_success",
        indicator_result={},
        node_statuses={},
        progress_events=[],
        raw_state={"requested_outputs": ["image"]},
        raw_submission={},
    )

    assert any(
        call["label"] == "Images" and call["expanded"]
        for call in dummy_st.expander_calls
    )
    assert any("Image Prompts" in call for call in dummy_st.markdown_calls)
    assert any(
        image[0] == "https://img.example/a.png" for image in dummy_st.image_calls
    )
    assert any("recoverable issue" in call.lower() for call in dummy_st.warning_calls)


def test_research_section_does_not_duplicate_blog_output(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "success",
        "final_response": (
            "## Blog Draft\nBlog content.\n\n"
            "## Research Summary / Research Report\nResearch content."
        ),
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=[],
        execution_status="success",
        indicator_result={},
        node_statuses={},
        progress_events=[],
        raw_state={"requested_outputs": ["blog", "research"]},
        raw_submission={},
    )

    assert any(call["label"] == "Research" for call in dummy_st.expander_calls)
    assert sum("Blog content." in call for call in dummy_st.markdown_calls) == 1
    assert sum("Research content." in call for call in dummy_st.markdown_calls) == 1


def test_debug_section_is_collapsed_and_sanitized(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "failed",
        "final_response": "",
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }
    raw_state = {
        "errors": [
            {
                "message": (
                    "Traceback (most recent call last): OPENAI_API_KEY=sk-secret"
                )
            }
        ]
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=[],
        execution_status="failed",
        indicator_result={},
        node_statuses={},
        progress_events=[],
        raw_state=raw_state,
        raw_submission={},
    )

    debug_calls = [
        call for call in dummy_st.expander_calls if call["label"] == "Debug / Advanced"
    ]
    assert debug_calls and debug_calls[0]["expanded"] is False
    assert dummy_st.json_calls
    debug_blob = str(dummy_st.json_calls[-1])
    assert "OPENAI_API_KEY" not in debug_blob
    assert "Traceback (most recent call last):" not in debug_blob


def test_collapsible_renderer_does_not_mutate_inputs(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    render_payload = {
        "workflow_status": "success",
        "final_response": "## Blog Draft\nBody",
        "usage_summary": {},
        "image_prompts": ["Prompt"],
        "image_outputs": [],
        "sources": [],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }
    status_messages = ["done"]
    node_statuses = {"query_handler_node": "completed"}
    progress_events = [{"node_name": "query_handler_node", "status": "completed"}]
    raw_state = {"requested_outputs": ["blog"]}
    raw_submission = {"requested_outputs": ["blog"]}

    before = copy.deepcopy(
        {
            "render_payload": render_payload,
            "status_messages": status_messages,
            "node_statuses": node_statuses,
            "progress_events": progress_events,
            "raw_state": raw_state,
            "raw_submission": raw_submission,
        }
    )

    result_view_module.render_collapsible_output_sections(
        render_payload=render_payload,
        status_messages=status_messages,
        execution_status="success",
        indicator_result={"workflow_status": "success"},
        node_statuses=node_statuses,
        progress_events=progress_events,
        raw_state=raw_state,
        raw_submission=raw_submission,
    )

    after = {
        "render_payload": render_payload,
        "status_messages": status_messages,
        "node_statuses": node_statuses,
        "progress_events": progress_events,
        "raw_state": raw_state,
        "raw_submission": raw_submission,
    }
    assert before == after
