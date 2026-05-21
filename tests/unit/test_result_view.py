from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List, Mapping

from frontend.components import result_view as result_view_module

_PROVIDER_WARNING = (
    "Text provider unavailable or quota-limited. "
    "ContentBlitz generated limited fallback outputs."
)
_TEXT_FALLBACK_WARNING = (
    "Text generation was unavailable. "
    "Blog and LinkedIn outputs are fallback outlines."
)
_IMAGE_DEGRADED_WARNING = (
    "Image generation encountered a recoverable issue. "
    "Text/research/export outputs remain available."
)


@dataclass
class _DummyContextManager:
    def __enter__(self) -> "_DummyContextManager":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


@dataclass
class _DummyStreamlit:
    subheader_calls: List[str] = field(default_factory=list)
    markdown_calls: List[str] = field(default_factory=list)
    image_calls: List[tuple[str, str]] = field(default_factory=list)
    info_calls: List[str] = field(default_factory=list)
    success_calls: List[str] = field(default_factory=list)
    warning_calls: List[str] = field(default_factory=list)
    error_calls: List[str] = field(default_factory=list)
    expander_calls: List[dict[str, Any]] = field(default_factory=list)
    json_calls: List[Any] = field(default_factory=list)
    json_kwargs: List[dict[str, Any]] = field(default_factory=list)
    download_button_calls: List[dict[str, Any]] = field(default_factory=list)

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        if _args:
            self.subheader_calls.append(str(_args[0]))
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
        self.json_kwargs.append(dict(_kwargs))

    def download_button(self, **kwargs: Any) -> None:
        self.download_button_calls.append(dict(kwargs))


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
        ("https://img.example/a.png", "Generated image (completed)")
    ]


def test_image_output_with_local_path_renders_streamlit_image(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)
    monkeypatch.delenv("CONTENTBLITZ_EXPORT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    image_dir = tmp_path / "exports" / "images"
    image_dir.mkdir(parents=True)
    image_file = image_dir / "example.png"
    image_file.write_bytes(b"png bytes")

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
        (str(image_file.resolve()), "Generated image (completed)")
    ]
    assert dummy_st.download_button_calls == [
        {
            "label": "Download Image",
            "data": b"png bytes",
            "file_name": "example.png",
            "mime": "image/png",
        }
    ]
    rendered_text = "\n".join(dummy_st.markdown_calls)
    assert "gpt-image-1" not in rendered_text
    assert "exports/images/example.png" not in rendered_text


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


def test_missing_local_image_output_renders_safe_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)
    monkeypatch.delenv("CONTENTBLITZ_EXPORT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    payload = _render_payload_with_images(
        [
            {
                "status": "success",
                "provider": "gpt-image-1",
                "local_path": "exports/images/missing.png",
                "renderable": True,
            }
        ]
    )
    result_view_module.render_partial_outputs(payload)

    assert dummy_st.image_calls == []
    assert dummy_st.download_button_calls == []
    assert dummy_st.warning_calls == [
        "Generated image file is not available for download."
    ]
    assert not any("missing.png" in call for call in dummy_st.warning_calls)


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


def test_blog_content_strips_trailing_sources_but_keeps_inline_links(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "success",
        "final_response": "",
        "partial_output_sections": [
            {
                "key": "blog",
                "label": "Blog Draft",
                "content": (
                    "Body with inline citation "
                    "[source](https://example.com/inline).\n\n"
                    "## Sources\n"
                    "[1] Source block item"
                ),
            }
        ],
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [
            {
                "title": "Structured Source",
                "url": "https://example.com/structured",
                "snippet": "structured snippet",
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
        raw_state={"requested_outputs": ["blog"]},
        raw_submission={},
    )

    assert any(
        "[source](https://example.com/inline)" in call
        for call in dummy_st.markdown_calls
    )
    assert not any("## Sources" in call for call in dummy_st.markdown_calls)
    assert not any("Source block item" in call for call in dummy_st.markdown_calls)


def test_final_response_with_duplicate_sources_is_trimmed_in_content_panels(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "success",
        "final_response": (
            "## Blog Draft\n"
            "Main blog content.\n\n"
            "## Sources\n"
            "[1] Duplicate source list one\n\n"
            "## Sources\n"
            "[1] Duplicate source list two"
        ),
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [
            {
                "title": "Structured Source",
                "url": "https://example.com/structured",
                "snippet": "structured snippet",
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
        raw_state={"requested_outputs": ["blog"]},
        raw_submission={},
    )

    assert any("Main blog content." in call for call in dummy_st.markdown_calls)
    assert not any(
        "Duplicate source list one" in call for call in dummy_st.markdown_calls
    )
    assert not any(
        "Duplicate source list two" in call for call in dummy_st.markdown_calls
    )
    assert sum(call["label"] == "Sources" for call in dummy_st.expander_calls) == 1


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


def test_execution_indicators_render_expected_summary_cards(
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
    assert any("Active Node" in call for call in dummy_st.markdown_calls)
    assert not any("Routing" in call for call in dummy_st.markdown_calls)
    assert not any("cbx-summary-dot-blue" in call for call in dummy_st.markdown_calls)
    assert any("cbx-summary-dot-idle" in call for call in dummy_st.markdown_calls)


def test_status_color_mapping_for_workflow_and_node_statuses(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_execution_indicators(
        execution_status="running",
        result={"ui_workflow_status": "success", "routing_decision": "done"},
    )
    result_view_module.render_result_header({"workflow_status": "partial_success"})
    result_view_module.render_result_header({"workflow_status": "limited"})
    result_view_module.render_node_execution_statuses(
        [
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
            {
                "node_name": "clarification_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:01+00:00",
            },
            {
                "node_name": "research_agent_node",
                "status": "skipped",
                "timestamp": "2026-05-17T12:00:02+00:00",
            },
            {
                "node_name": "content_strategist_node",
                "status": "failed",
                "timestamp": "2026-05-17T12:00:03+00:00",
            },
            {
                "node_name": "blog_writer_node",
                "status": "degraded",
                "timestamp": "2026-05-17T12:00:04+00:00",
            },
        ]
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "cbx-status-green" in rendered
    assert "cbx-status-orange" in rendered
    assert "cbx-status-red" in rendered
    assert "limited" in rendered
    assert "partial success" in rendered
    assert "partial_success" not in rendered
    assert "cbx-node-status-list" in rendered
    assert "cbx-node-progress-track" in rendered
    assert "cbx-node-progress-fill" in rendered
    assert "cbx-node-progress-running" in rendered


def test_static_uninvoked_nodes_are_not_rendered(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_node_execution_statuses(
        [
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ]
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "query_handler_node" in rendered
    assert "clarification_node" not in rendered
    assert "research_agent_node" not in rendered


def test_seeded_pending_events_do_not_render_uninvoked_nodes(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_node_execution_statuses(
        [
            {
                "node_name": "query_handler_node",
                "status": "pending",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
            {
                "node_name": "research_agent_node",
                "status": "pending",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
        ]
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "query_handler_node" not in rendered
    assert "research_agent_node" not in rendered


def test_running_node_renders_immediately_with_in_progress_style(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_node_execution_statuses(
        [
            {
                "node_name": "query_handler_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ]
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "query_handler_node" in rendered
    assert "◎" in rendered
    assert "cbx-node-status-running" in rendered
    assert "cbx-node-progress-running" in rendered


def test_execution_indicators_show_active_running_node(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_execution_indicators(
        execution_status="running",
        result={
            "ui_workflow_status": "running",
            "routing_decision": "research_agent_node",
        },
        progress_events=[
            {
                "node_name": "research_agent_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ],
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "Active Node" in rendered
    assert "research_agent_node" in rendered
    assert "cbx-summary-dot-running" in rendered
    assert "Routing" not in rendered
    assert "cbx-summary-dot-blue" not in rendered


def test_execution_indicators_support_completed_dot_state(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_execution_indicators(
        execution_status="completed",
        result={
            "ui_workflow_status": "success",
            "routing_decision": "output_assembler_node",
        },
        progress_events=[
            {
                "node_name": "output_assembler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ],
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "cbx-summary-dot-completed" in rendered
    assert "output_assembler_node" in rendered


def test_node_execution_rows_follow_first_invoked_order_and_latest_status() -> None:
    rows = result_view_module._build_node_execution_rows(
        [
            {
                "node_name": "research_agent_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:02+00:00",
            },
            {
                "node_name": "query_handler_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:01+00:00",
            },
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:03+00:00",
            },
            {
                "node_name": "research_agent_node",
                "status": "degraded",
                "timestamp": "2026-05-17T12:00:04+00:00",
            },
        ]
    )

    assert [row.node_name for row in rows] == [
        "research_agent_node",
        "query_handler_node",
    ]
    assert [row.status for row in rows] == ["degraded", "completed"]


def test_node_duration_weights_match_expected_relative_costs() -> None:
    assert result_view_module._node_duration_seconds(
        "research_agent_node"
    ) > result_view_module._node_duration_seconds("blog_writer_node")
    assert result_view_module._node_duration_seconds(
        "blog_writer_node"
    ) > result_view_module._node_duration_seconds("output_assembler_node")
    assert (
        result_view_module._node_duration_seconds("export_node")
        == result_view_module._node_duration_seconds("output_assembler_node")
    )


def test_pipeline_elapsed_helper_uses_meaningful_event_timestamps() -> None:
    elapsed = result_view_module._pipeline_elapsed_label_from_events(
        [
            {
                "node_name": "query_handler_node",
                "status": "pending",
                "timestamp": "2026-05-17T11:59:00+00:00",
            },
            {
                "node_name": "query_handler_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:02+00:00",
            },
        ]
    )

    assert elapsed == "2.0s"


def test_pipeline_elapsed_helper_uses_active_now_for_running_events() -> None:
    elapsed = result_view_module._pipeline_elapsed_label_from_events(
        [
            {
                "node_name": "query_handler_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ],
        now=datetime(2026, 5, 17, 12, 0, 5, tzinfo=UTC),
    )

    assert elapsed == "5.0s"


def test_running_node_elapsed_starts_at_zero_without_live_now() -> None:
    rows = result_view_module._build_node_execution_rows(
        [
            {
                "node_name": "query_handler_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ]
    )

    assert rows[0].elapsed_label == "0.0s"


def test_completed_node_elapsed_uses_start_and_end_timestamps() -> None:
    rows = result_view_module._build_node_execution_rows(
        [
            {
                "node_name": "query_handler_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:02+00:00",
            },
        ]
    )

    assert rows[0].elapsed_label == "2.0s"


def test_completed_node_elapsed_falls_back_to_weighted_duration_when_zero() -> None:
    rows = result_view_module._build_node_execution_rows(
        [
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ]
    )

    assert rows[0].elapsed_label == "1.2s"


def test_completed_node_elapsed_does_not_shift_with_later_node_events() -> None:
    rows = result_view_module._build_node_execution_rows(
        [
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
            {
                "node_name": "research_agent_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:05+00:00",
            },
        ]
    )

    by_node = {row.node_name: row.elapsed_label for row in rows}
    assert by_node["query_handler_node"] == "1.2s"
    assert by_node["research_agent_node"] == "3.2s"


def test_skipped_nodes_are_hidden_from_node_execution_status(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_node_execution_statuses(
        [
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
            {
                "node_name": "export_node",
                "status": "skipped",
                "timestamp": "2026-05-17T12:00:01+00:00",
            },
        ]
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "query_handler_node" in rendered
    assert "export_node" not in rendered
    assert "image_agent_node" not in rendered


def test_node_execution_rows_keep_first_meaningful_event_order() -> None:
    rows = result_view_module._build_node_execution_rows(
        [
            {
                "node_name": "clarification_node",
                "status": "pending",
                "timestamp": "2026-05-17T11:59:58+00:00",
            },
            {
                "node_name": "query_handler_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:00+00:00",
            },
            {
                "node_name": "blog_writer_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:01+00:00",
            },
            {
                "node_name": "linkedin_writer_node",
                "status": "running",
                "timestamp": "2026-05-17T12:00:01+00:00",
            },
            {
                "node_name": "blog_writer_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:02+00:00",
            },
            {
                "node_name": "linkedin_writer_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:03+00:00",
            },
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:04+00:00",
            },
        ]
    )

    assert [row.node_name for row in rows] == [
        "query_handler_node",
        "blog_writer_node",
        "linkedin_writer_node",
    ]
    assert "clarification_node" not in [row.node_name for row in rows]


def test_completed_node_row_uses_horizontal_layout_and_right_status_badge(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_node_execution_statuses(
        [
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "timestamp": "2026-05-17T12:00:00+00:00",
            }
        ]
    )

    rendered = "\n".join(dummy_st.markdown_calls)
    assert "cbx-node-status-row" in rendered
    assert "cbx-node-progress-track" in rendered
    assert "cbx-node-status-badge" in rendered
    assert "✓" in rendered
    assert "Elapsed 0.0s" in rendered
    assert "1.2s" in rendered
    assert "width:100%;" in rendered
    assert (
        rendered.find("cbx-node-name")
        < rendered.find("cbx-node-progress-track")
        < rendered.find("cbx-node-status-badge")
    )
    assert "Parallel Group" not in rendered


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


def test_performance_summary_renders_safe_cards(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    result_view_module.render_performance_summary(
        {
            "performance_summary": {
                "executed_node_count": 4,
                "timed_node_count": 4,
                "total_duration_ms": 2200,
                "average_duration_ms": 550,
                "provider_latency_total_ms": 1400,
                "nodes": [
                    {
                        "node_name": "query_handler_node",
                        "status": "completed",
                        "duration_ms": 200,
                    },
                    {
                        "node_name": "research_agent_node",
                        "status": "completed",
                        "duration_ms": 1200,
                        "provider": "serp_api",
                        "provider_latency_ms": 1200,
                        "cache_hit": False,
                    },
                ],
            }
        }
    )

    assert "Performance Summary" in dummy_st.subheader_calls
    assert any("Executed Nodes" in call for call in dummy_st.markdown_calls)
    assert any("Total Duration" in call for call in dummy_st.markdown_calls)


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
    assert any("cbx-status-green" in call for call in dummy_st.markdown_calls)
    assert any("completed" in call for call in dummy_st.markdown_calls)
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


def test_linkedin_panel_renders_duplicated_first_paragraph_once(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    first_paragraph = "First paragraph should only render once."
    payload = {
        "workflow_status": "success",
        "final_response": (
            "## LinkedIn Post\n"
            f"{first_paragraph}\n\n"
            f"{first_paragraph}\n\n"
            "Read the full piece: [details](https://example.com/post)\n\n"
            "#ContentMarketing #AI"
        ),
        "partial_output_sections": [
            {
                "key": "linkedin",
                "label": "LinkedIn Draft",
                "content": (
                    f"{first_paragraph}\n\n"
                    "Read the full piece: [details](https://example.com/post)"
                ),
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
        raw_state={"requested_outputs": ["linkedin"]},
        raw_submission={},
    )

    rendered_text = "\n".join(dummy_st.markdown_calls)
    assert rendered_text.count(first_paragraph) == 1
    assert "[details](https://example.com/post)" in rendered_text
    assert "#ContentMarketing #AI" in rendered_text
    assert sum(call["label"] == "LinkedIn" for call in dummy_st.expander_calls) == 1
    assert not any(call["label"] == "Research" for call in dummy_st.expander_calls)


def test_sectioned_sources_render_once_as_clickable_list(monkeypatch) -> None:
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
                "source": "serp_api",
            },
            {
                "title": "Source Without URL",
                "url": "",
                "snippet": "Snippet B",
            },
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
    source_calls = [call for call in dummy_st.markdown_calls if "Source A" in call]
    assert len(source_calls) == 1
    assert "1. [Source A](https://example.com/a) - Snippet A" in source_calls[0]
    assert "2. Source Without URL - Snippet B" in source_calls[0]
    assert "Snippet B" in source_calls[0]
    assert "serp_api" not in source_calls[0]
    assert not any("cbx-source-card" in call for call in dummy_st.markdown_calls)


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
        call["label"] == "Images" and not call["expanded"]
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


def test_observability_section_renders_above_workflow_messages(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)
    monkeypatch.setattr(
        result_view_module,
        "build_observability_diagnostics",
        lambda: {
            "status": "enabled",
            "status_label": "Enabled",
            "tracing_enabled": True,
            "project_name": "ContentBlitz",
            "endpoint_host": "api.smith.langchain.com",
            "last_trace_attempt_label": "Ready",
            "note": "Tracing is enabled for this app session.",
            "dashboard_instruction": (
                "For trace details, review the LangSmith dashboard manually."
            ),
        },
    )

    payload = {
        "workflow_status": "success",
        "final_response": "## Blog Draft\nBody",
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=["Workflow completed successfully."],
        execution_status="success",
        indicator_result={},
        node_statuses={},
        progress_events=[],
        raw_state={"requested_outputs": ["blog"]},
        raw_submission={},
    )

    assert dummy_st.subheader_calls.count("Observability") == 1
    assert dummy_st.subheader_calls.count("Workflow Messages") == 1
    assert dummy_st.subheader_calls.index(
        "Observability"
    ) < dummy_st.subheader_calls.index(
        "Workflow Messages",
    )
    rendered_markdown = "\n".join(dummy_st.markdown_calls)
    assert "Tracing Enabled" in rendered_markdown
    assert "Endpoint Host" in rendered_markdown


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
    assert any("Raw Workflow State" in call for call in dummy_st.markdown_calls)
    assert any("Render Payload" in call for call in dummy_st.markdown_calls)
    assert dummy_st.json_kwargs == [
        {"expanded": True},
        {"expanded": False},
        {"expanded": False},
    ]
    assert dummy_st.json_calls
    debug_blob = str(dummy_st.json_calls[-1])
    assert "OPENAI_API_KEY" not in debug_blob
    assert "Traceback (most recent call last):" not in debug_blob


def test_export_artifacts_render_download_buttons(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)
    monkeypatch.delenv("CONTENTBLITZ_EXPORT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    export_file = export_dir / "content_test.md"
    export_file.write_text("# Exported content", encoding="utf-8")

    payload = {
        "export_status": {
            "requested": True,
            "paths": {"markdown": "exports/content_test.md"},
            "errors": [],
        },
    }

    result_view_module.render_export_status(payload)

    assert dummy_st.download_button_calls == [
        {
            "label": "Download Markdown",
            "data": b"# Exported content",
            "file_name": "content_test.md",
            "mime": "text/markdown",
        }
    ]
    assert not any(
        "exports/content_test.md" in call for call in dummy_st.markdown_calls
    )


def test_missing_export_file_renders_safe_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)
    monkeypatch.delenv("CONTENTBLITZ_EXPORT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    payload = {
        "export_status": {
            "requested": True,
            "paths": {"pdf": "exports/missing.pdf"},
            "errors": [],
        },
    }

    result_view_module.render_export_status(payload)

    assert dummy_st.download_button_calls == []
    assert dummy_st.warning_calls == [
        "PDF export file is not available for download."
    ]
    assert not any("exports/missing.pdf" in call for call in dummy_st.warning_calls)


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


def test_provider_degradation_banner_and_status_cards_render(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "partial_success",
        "final_response": "## Blog Draft\nFallback body.",
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "provider_status": {
            "text_generation": "degraded",
            "image_generation": "completed",
            "search": "completed",
            "export": "completed",
        },
        "degradation_metadata": {
            "text_generation_degraded": True,
            "image_generation_degraded": False,
            "fallback_content_used": True,
            "real_generation_succeeded": False,
            "provider_failure_reason": "quota_exceeded",
        },
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=[],
        execution_status="partial_success",
        indicator_result={"workflow_status": "partial_success"},
        node_statuses={"blog_writer_node": "completed"},
        progress_events=[{"node_name": "blog_writer_node", "status": "completed"}],
        raw_state={"requested_outputs": ["blog"]},
        raw_submission={},
    )

    assert any(
        "Text provider unavailable or quota-limited. "
        "ContentBlitz generated limited fallback outputs."
        in call
        for call in dummy_st.warning_calls
    )
    assert any("Text Generation" in call for call in dummy_st.markdown_calls)


def test_provider_warning_messages_are_deduped_in_workflow_view(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "partial_success",
        "final_response": "## Blog Draft\nFallback body.",
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "provider_status": {
            "text_generation": "degraded",
            "image_generation": "degraded",
            "search": "completed",
            "export": "completed",
        },
        "degradation_metadata": {
            "text_generation_degraded": True,
            "image_generation_degraded": True,
            "fallback_content_used": True,
            "real_generation_succeeded": False,
            "provider_failure_reason": "authentication_failed",
        },
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    status_messages = [
        _PROVIDER_WARNING,
        (
            "Draft unavailable because text generation is currently limited. "
            "Research sources were collected successfully and can be used to "
            "regenerate this section once the provider is available."
        ),
        _IMAGE_DEGRADED_WARNING,
        _PROVIDER_WARNING,
    ]
    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=status_messages,
        execution_status="partial_success",
        indicator_result={"workflow_status": "partial_success"},
        node_statuses={"blog_writer_node": "degraded"},
        progress_events=[{"node_name": "blog_writer_node", "status": "degraded"}],
        raw_state={"requested_outputs": ["blog", "linkedin", "image"]},
        raw_submission={},
    )

    assert (
        dummy_st.warning_calls.count(_PROVIDER_WARNING)
        == 1
    )
    assert dummy_st.info_calls.count(_TEXT_FALLBACK_WARNING) == 1
    assert dummy_st.info_calls.count(_IMAGE_DEGRADED_WARNING) == 1


def test_blog_section_shows_fallback_badges_when_degraded(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(result_view_module, "st", dummy_st)

    payload = {
        "workflow_status": "partial_success",
        "final_response": "## Blog Draft\nFallback body.",
        "usage_summary": {},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "degradation_metadata": {
            "text_generation_degraded": True,
            "image_generation_degraded": False,
        },
        "provider_status": {},
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }

    result_view_module.render_collapsible_output_sections(
        render_payload=payload,
        status_messages=[],
        execution_status="partial_success",
        indicator_result={"workflow_status": "partial_success"},
        node_statuses={"blog_writer_node": "completed"},
        progress_events=[{"node_name": "blog_writer_node", "status": "completed"}],
        raw_state={"requested_outputs": ["blog"]},
        raw_submission={},
    )

    rendered_markup = "\n".join(dummy_st.markdown_calls).lower()
    assert "provider degraded" in rendered_markup
    assert "limited" in rendered_markup
