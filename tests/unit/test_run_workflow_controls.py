from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from frontend.pages import run_workflow as run_workflow_page


@dataclass
class _DummyContextManager:
    def __enter__(self) -> "_DummyContextManager":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


@dataclass
class _DummyPlaceholder:
    parent: "_DummyStreamlit"

    def container(self) -> _DummyContextManager:
        return _DummyContextManager()

    def empty(self) -> None:
        self.parent.placeholder_empty_calls += 1


@dataclass
class _DummyStreamlit:
    session_state: Dict[str, Any] = field(default_factory=dict)
    text_area_value: str = "Create a blog and export as PDF."
    button_result: bool = True
    caption_calls: List[str] = field(default_factory=list)
    markdown_calls: List[str] = field(default_factory=list)
    info_calls: List[str] = field(default_factory=list)
    placeholder_empty_calls: int = 0

    def header(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def caption(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.caption_calls.append(str(value))

    def markdown(self, value: Any, *_args: Any, **_kwargs: Any) -> None:
        self.markdown_calls.append(str(value))

    def container(self, *_args: Any, **_kwargs: Any) -> _DummyContextManager:
        return _DummyContextManager()

    def text_area(self, *_args: Any, **_kwargs: Any) -> str:
        return self.text_area_value

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        return self.button_result

    def pills(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def empty(self) -> _DummyPlaceholder:
        return _DummyPlaceholder(parent=self)

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def info(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.info_calls.append(str(value))

    def expander(self, *_args: Any, **_kwargs: Any) -> _DummyContextManager:
        return _DummyContextManager()

    def json(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def test_validation_requires_prompt_only() -> None:
    error = run_workflow_page._validate_submission_inputs(safe_query="")
    assert error == "A prompt is required before running the workflow."


def test_render_uses_prompt_only_submission_and_backend_selected_options(
    monkeypatch,
) -> None:
    dummy_st = _DummyStreamlit(
        text_area_value="Create a blog article and export as html and docx."
    )
    monkeypatch.setattr(run_workflow_page, "st", dummy_st)

    stream_calls: List[Dict[str, Any]] = []
    history_calls: List[Dict[str, Any]] = []
    progress_event = {
        "node_name": "export_node",
        "status": "degraded",
        "message": "export_node completed with warnings.",
        "timestamp": "2026-05-16T12:00:00+00:00",
    }
    final_result = {
        "workflow_status": "success",
        "final_response": "## Blog Draft\nFinal assembled blog.",
        "content_drafts": {"blog": {"body": "Final assembled blog."}},
        "research_data": {"degraded": False},
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "warnings": [],
        "errors": [],
        "quality_scores": {},
        "requested_outputs": ["blog"],
        "export_requested": True,
        "export_metadata": {
            "formats_requested": ["html", "docx"],
            "export_paths": {},
            "error_log": [],
        },
        "cost_controls": {"budget_exceeded": False},
    }

    def _fake_stream_workflow_progress(**kwargs: Any):
        stream_calls.append(dict(kwargs))
        yield {"type": "progress", "event": dict(progress_event)}
        yield {
            "type": "final",
            "result": dict(final_result),
            "events": [dict(progress_event)],
        }

    monkeypatch.setattr(
        run_workflow_page,
        "stream_workflow_progress",
        _fake_stream_workflow_progress,
    )

    store: Dict[str, Any] = {
        "execution_status": "idle",
        "last_error": "",
        "last_result": None,
        "progress_events": [],
        "node_statuses": {},
        "status_messages": [],
        "last_submission": {},
        "persistence_messages": [],
    }
    monkeypatch.setattr(
        run_workflow_page,
        "set_last_submission",
        lambda value: store.update({"last_submission": value}),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "set_execution_status",
        lambda value: store.update({"execution_status": value}),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "set_last_error",
        lambda value: store.update({"last_error": value}),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "set_progress_events",
        lambda value: store.update({"progress_events": list(value)}),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "set_node_statuses",
        lambda value: store.update({"node_statuses": dict(value)}),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "set_status_messages",
        lambda value: store.update({"status_messages": list(value)}),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "set_last_result",
        lambda value: store.update({"last_result": dict(value)}),
    )
    monkeypatch.setattr(run_workflow_page, "save_persisted_run", lambda **_kwargs: "")
    monkeypatch.setattr(
        run_workflow_page,
        "add_history_entry",
        lambda **kwargs: history_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        run_workflow_page, "get_last_error", lambda: store["last_error"]
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_persistence_messages",
        lambda: list(store["persistence_messages"]),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "clear_persistence_messages",
        lambda: store.update({"persistence_messages": []}),
    )
    monkeypatch.setattr(
        run_workflow_page, "get_last_result", lambda: store["last_result"]
    )
    monkeypatch.setattr(
        run_workflow_page, "get_execution_status", lambda: store["execution_status"]
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_progress_events",
        lambda: list(store["progress_events"]),
    )
    monkeypatch.setattr(
        run_workflow_page, "get_node_statuses", lambda: dict(store["node_statuses"])
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_status_messages",
        lambda: list(store["status_messages"]),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_last_submission",
        lambda: dict(store["last_submission"]),
    )

    monkeypatch.setattr(
        run_workflow_page,
        "render_collapsible_output_sections",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_workflow_page, "render_degraded_and_error_state", lambda _payload: None
    )
    monkeypatch.setattr(
        run_workflow_page,
        "_render_active_execution_state",
        lambda **_kwargs: None,
    )

    run_workflow_page.render()

    assert stream_calls == [{"user_query": dummy_st.text_area_value.strip()}]
    assert "requested_outputs" not in stream_calls[0]
    assert "export_requested" not in stream_calls[0]
    assert "export_formats" not in stream_calls[0]

    assert store["last_submission"]["requested_outputs"] == ["blog"]
    assert store["last_submission"]["export_requested"] is True
    assert store["last_submission"]["export_formats"] == ["html", "docx"]
    assert store["last_submission"]["user_query"] == dummy_st.text_area_value.strip()
    assert len(history_calls) == 1
    assert history_calls[0]["workflow_status"] == "partial_success"


def test_active_submitted_run_uses_user_query_submission_flag(monkeypatch) -> None:
    dummy_st = _DummyStreamlit(button_result=False)
    monkeypatch.setattr(run_workflow_page, "st", dummy_st)

    store: Dict[str, Any] = {
        "execution_status": "running",
        "last_error": "",
        "last_result": None,
        "progress_events": [],
        "node_statuses": {},
        "status_messages": ["Workflow started."],
        "last_submission": {"user_query": "create a workflow"},
        "persistence_messages": [],
    }
    monkeypatch.setattr(
        run_workflow_page, "get_last_error", lambda: store["last_error"]
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_persistence_messages",
        lambda: list(store["persistence_messages"]),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "clear_persistence_messages",
        lambda: store.update({"persistence_messages": []}),
    )
    monkeypatch.setattr(
        run_workflow_page, "get_last_result", lambda: store["last_result"]
    )
    monkeypatch.setattr(
        run_workflow_page, "get_execution_status", lambda: store["execution_status"]
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_progress_events",
        lambda: list(store["progress_events"]),
    )
    monkeypatch.setattr(
        run_workflow_page, "get_node_statuses", lambda: dict(store["node_statuses"])
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_status_messages",
        lambda: list(store["status_messages"]),
    )
    monkeypatch.setattr(
        run_workflow_page,
        "get_last_submission",
        lambda: dict(store["last_submission"]),
    )

    live_state_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        run_workflow_page,
        "_render_active_execution_state",
        lambda **kwargs: live_state_calls.append(dict(kwargs)),
    )

    run_workflow_page.render()

    assert len(live_state_calls) == 1
    assert live_state_calls[0]["execution_status"] == "running"
    assert dummy_st.info_calls == []


def test_prompt_note_mentions_backend_deterministic_outputs(monkeypatch) -> None:
    dummy_st = _DummyStreamlit(button_result=False)
    monkeypatch.setattr(run_workflow_page, "st", dummy_st)

    monkeypatch.setattr(run_workflow_page, "get_last_error", lambda: "")
    monkeypatch.setattr(run_workflow_page, "get_persistence_messages", lambda: [])
    monkeypatch.setattr(run_workflow_page, "clear_persistence_messages", lambda: None)
    monkeypatch.setattr(run_workflow_page, "get_last_result", lambda: None)
    monkeypatch.setattr(run_workflow_page, "get_execution_status", lambda: "idle")
    monkeypatch.setattr(run_workflow_page, "get_progress_events", lambda: [])
    monkeypatch.setattr(run_workflow_page, "get_node_statuses", lambda: {})
    monkeypatch.setattr(run_workflow_page, "get_status_messages", lambda: [])
    monkeypatch.setattr(run_workflow_page, "get_last_submission", lambda: {})

    run_workflow_page.render()

    assert any(
        "Generate blogs, LinkedIn posts, research reports, image concepts,"
        in message
        for message in dummy_st.markdown_calls
    )
