from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from frontend.pages import history as history_page


@dataclass
class _DummyButtonColumn:
    button_result: bool = False

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        return self.button_result

    def caption(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass
class _DummyCaptionColumn:
    def caption(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass
class _DummyStreamlit:
    warnings: List[str] = field(default_factory=list)

    def header(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, message: str) -> None:
        self.warnings.append(str(message))

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def selectbox(
        self,
        _label: str,
        options: List[str],
        format_func=None,
    ) -> str:
        if not options:
            return ""
        selected = str(options[0])
        if format_func is not None:
            _ = format_func(selected)
        return selected

    def columns(self, _spec) -> tuple[_DummyButtonColumn, _DummyCaptionColumn]:
        return _DummyButtonColumn(button_result=False), _DummyCaptionColumn()

    def divider(self) -> None:
        return None

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def markdown(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def success(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def test_history_page_renders_usage_summary_for_loaded_run(monkeypatch) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(history_page, "st", dummy_st)

    monkeypatch.setattr(history_page, "get_persistence_messages", lambda: [])
    monkeypatch.setattr(history_page, "clear_persistence_messages", lambda: None)
    monkeypatch.setattr(
        history_page,
        "list_persisted_run_summaries",
        lambda limit=200: [
            {
                "run_id": "run_1",
                "updated_at": "2026-05-12T00:00:00+00:00",
                "workflow_status": "partial_success",
                "requested_outputs": ["blog"],
                "user_query_preview": "Create a short AI blog post.",
            }
        ],
    )
    monkeypatch.setattr(
        history_page,
        "load_persisted_run",
        lambda run_id: {
            "run_id": run_id,
            "session_id": "session_1",
            "status_messages": [],
            "ui_node_statuses": {},
        },
    )
    monkeypatch.setattr(history_page, "restore_persisted_run", lambda _run_id: (True, "ok"))
    monkeypatch.setattr(history_page, "get_run_history", lambda: [])

    render_payload: Dict[str, Any] = {
        "workflow_status": "partial_success",
        "final_response": "Saved result",
        "warnings": [],
        "errors": [],
        "sources": [],
        "image_prompts": [],
        "image_outputs": [],
        "partial_output_mode": "none",
        "partial_output_sections": [],
        "usage_summary": {
            "estimated_tokens_in": 1200,
            "estimated_tokens_out": 3000,
            "search_queries": 3,
            "sources_returned": 5,
            "image_generation_requests": 1,
            "image_generation_failures": 0,
            "retry_attempts": 0,
            "degraded_operations": 1,
            "export_generation_count": 0,
            "estimated_total_operations": 6,
            "estimated_workflow_cost_level": "medium",
            "budget_state": "degraded",
        },
        "export_status": {"requested": False, "paths": {}, "errors": []},
    }
    monkeypatch.setattr(
        history_page,
        "build_render_payload",
        lambda *, state, node_statuses: render_payload,
    )

    monkeypatch.setattr(history_page, "render_status_messages", lambda _messages: None)
    monkeypatch.setattr(history_page, "render_degraded_and_error_state", lambda _payload: None)
    monkeypatch.setattr(history_page, "render_partial_outputs", lambda _payload: None)
    monkeypatch.setattr(history_page, "render_result_header", lambda _result: None)
    monkeypatch.setattr(history_page, "render_final_response", lambda _result: None)
    monkeypatch.setattr(history_page, "render_sources", lambda _result: None)
    monkeypatch.setattr(history_page, "render_export_status", lambda _payload: None)

    captured: Dict[str, Any] = {}

    def _capture_usage(payload: Mapping[str, Any]) -> None:
        captured.update(payload)

    monkeypatch.setattr(history_page, "render_usage_summary", _capture_usage)

    history_page.render()

    assert "usage_summary" in captured
    usage_summary = captured["usage_summary"]
    assert isinstance(usage_summary, dict)
    assert usage_summary["search_queries"] == 3
    assert usage_summary["estimated_tokens_out"] == 3000
