from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from frontend.pages import run_workflow as run_workflow_page
from frontend.services.submission_options import WorkflowControls


@dataclass
class _DummyColumn:
    parent: "_DummyStreamlit"

    def checkbox(
        self,
        label: str,
        value: bool = False,
        help: str | None = None,
        key: str | None = None,
    ) -> bool:
        return self.parent.checkbox(label, value=value, help=help, key=key)


@dataclass
class _DummyStreamlit:
    session_state: Dict[str, Any] = field(default_factory=dict)
    checkbox_values: Dict[str, bool] = field(default_factory=dict)
    multiselect_values: List[str] = field(default_factory=list)
    checkbox_calls: List[Dict[str, Any]] = field(default_factory=list)
    multiselect_calls: List[Dict[str, Any]] = field(default_factory=list)

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def caption(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def columns(self, count: int):
        return tuple(_DummyColumn(self) for _ in range(count))

    def checkbox(
        self,
        label: str,
        value: bool = False,
        help: str | None = None,
        key: str | None = None,
    ) -> bool:
        self.checkbox_calls.append(
            {
                "label": label,
                "value": value,
                "help": help,
                "key": key,
            }
        )
        return bool(self.checkbox_values.get(label, value))

    def multiselect(
        self,
        label: str,
        options: List[str],
        default: List[str] | None = None,
        key: str | None = None,
        help: str | None = None,
        disabled: bool = False,
    ) -> List[str]:
        self.multiselect_calls.append(
            {
                "label": label,
                "options": list(options),
                "default": list(default or []),
                "key": key,
                "help": help,
                "disabled": disabled,
            }
        )
        return list(self.multiselect_values)


def test_export_formats_control_remains_interactive_when_export_is_enabled(
    monkeypatch,
) -> None:
    dummy = _DummyStreamlit(
        checkbox_values={
            "Enable Export": True,
            "Blog Output": True,
            "LinkedIn Output": True,
            "Research Output": False,
            "Image Output": False,
        },
        multiselect_values=["pdf"],
    )
    monkeypatch.setattr(run_workflow_page, "st", dummy)

    controls = run_workflow_page._build_controls()

    assert controls.export_enabled is True
    assert controls.export_formats == ["pdf"]
    assert dummy.multiselect_calls
    assert dummy.multiselect_calls[0]["disabled"] is False


def test_export_formats_control_is_disabled_when_export_is_off(monkeypatch) -> None:
    dummy = _DummyStreamlit(
        checkbox_values={
            "Enable Export": False,
            "Blog Output": True,
            "LinkedIn Output": True,
            "Research Output": False,
            "Image Output": False,
        },
        multiselect_values=[],
    )
    monkeypatch.setattr(run_workflow_page, "st", dummy)

    controls = run_workflow_page._build_controls()

    assert controls.export_enabled is False
    assert controls.export_formats == []
    assert dummy.multiselect_calls
    assert dummy.multiselect_calls[0]["disabled"] is True
    assert dummy.multiselect_calls[0]["default"] == []


def test_export_formats_are_sanitized_in_controls(monkeypatch) -> None:
    dummy = _DummyStreamlit(
        checkbox_values={
            "Enable Export": True,
            "Blog Output": True,
            "LinkedIn Output": True,
            "Research Output": False,
            "Image Output": False,
        },
        multiselect_values=[" PDF ", "html", "pdf", "  "],
    )
    monkeypatch.setattr(run_workflow_page, "st", dummy)

    controls = run_workflow_page._build_controls()

    assert controls.export_enabled is True
    assert controls.export_formats == ["pdf", "html"]


def test_disabling_export_clears_multiselect_state(monkeypatch) -> None:
    dummy = _DummyStreamlit(
        session_state={
            run_workflow_page._WIDGET_KEY_EXPORT_FORMATS: ["markdown", "pdf"],
        },
        checkbox_values={
            "Enable Export": False,
            "Blog Output": True,
            "LinkedIn Output": True,
            "Research Output": False,
            "Image Output": False,
        },
        multiselect_values=[],
    )
    monkeypatch.setattr(run_workflow_page, "st", dummy)

    controls = run_workflow_page._build_controls()

    assert controls.export_enabled is False
    assert controls.export_formats == []
    assert dummy.session_state[run_workflow_page._WIDGET_KEY_EXPORT_FORMATS] == []


def test_validation_requires_export_format_when_export_is_enabled() -> None:
    error = run_workflow_page._validate_submission_inputs(
        safe_query="create a blog post",
        requested_outputs=["blog"],
        export_requested=True,
        export_formats=[],
    )
    assert error == "Select at least one export format when Enable Export is checked."


def test_validation_accepts_export_enabled_with_formats() -> None:
    error = run_workflow_page._validate_submission_inputs(
        safe_query="create a blog post",
        requested_outputs=["blog"],
        export_requested=True,
        export_formats=["markdown"],
    )
    assert error == ""


@dataclass
class _DummyContextManager:
    def __enter__(self) -> "_DummyContextManager":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


@dataclass
class _DummyPlaceholder:
    parent: "_DummyRenderStreamlit"

    def container(self) -> _DummyContextManager:
        return _DummyContextManager()

    def empty(self) -> None:
        self.parent.placeholder_empty_calls += 1


@dataclass
class _DummyRenderStreamlit:
    session_state: Dict[str, Any] = field(default_factory=dict)
    button_result: bool = True
    info_calls: List[str] = field(default_factory=list)
    caption_calls: List[str] = field(default_factory=list)
    spinner_calls: List[str] = field(default_factory=list)
    placeholder_empty_calls: int = 0

    def header(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def subheader(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def markdown(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def caption(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.caption_calls.append(str(value))

    def container(self, *_args: Any, **_kwargs: Any) -> _DummyContextManager:
        return _DummyContextManager()

    def empty(self) -> _DummyPlaceholder:
        return _DummyPlaceholder(parent=self)

    def text_area(self, *_args: Any, **_kwargs: Any) -> str:
        return "Create a workflow result."

    def button(self, *_args: Any, **_kwargs: Any) -> bool:
        return self.button_result

    def spinner(
        self,
        text: str = "",
        *_args: Any,
        **_kwargs: Any,
    ) -> _DummyContextManager:
        self.spinner_calls.append(str(text))
        return _DummyContextManager()

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


def test_node_execution_status_section_is_rendered_once_per_run(monkeypatch) -> None:
    dummy_st = _DummyRenderStreamlit()
    monkeypatch.setattr(run_workflow_page, "st", dummy_st)

    monkeypatch.setattr(
        run_workflow_page,
        "_build_controls",
        lambda: WorkflowControls(
            include_blog=True,
            include_linkedin=False,
            include_research=False,
            include_image=False,
            export_enabled=False,
            export_formats=[],
        ),
    )

    progress_event = {
        "node_name": "query_handler_node",
        "status": "completed",
        "message": "query_handler_node completed.",
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
        "export_requested": False,
        "export_metadata": {
            "formats_requested": [],
            "export_paths": {},
            "error_log": [],
        },
        "cost_controls": {"budget_exceeded": False},
        "requested_outputs": ["blog"],
    }

    def _fake_stream_workflow_progress(**_kwargs: Any):
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
    monkeypatch.setattr(run_workflow_page, "add_history_entry", lambda **_kwargs: None)
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
        run_workflow_page, "get_progress_events", lambda: list(store["progress_events"])
    )
    monkeypatch.setattr(
        run_workflow_page, "get_node_statuses", lambda: dict(store["node_statuses"])
    )
    monkeypatch.setattr(
        run_workflow_page, "get_status_messages", lambda: list(store["status_messages"])
    )
    monkeypatch.setattr(
        run_workflow_page, "get_last_submission", lambda: dict(store["last_submission"])
    )

    section_render_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        run_workflow_page,
        "render_collapsible_output_sections",
        lambda **kwargs: section_render_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        run_workflow_page, "render_degraded_and_error_state", lambda _payload: None
    )
    live_state_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        run_workflow_page,
        "_render_active_execution_state",
        lambda **kwargs: live_state_calls.append(
            {
                "execution_status": kwargs["execution_status"],
                "progress_events": list(kwargs["progress_events"]),
            }
        ),
    )

    run_workflow_page.render()

    assert len(section_render_calls) == 1
    assert (
        section_render_calls[0]["node_statuses"].get("query_handler_node")
        == "completed"
    )
    assert live_state_calls
    assert live_state_calls[0]["execution_status"] == "running"
    assert live_state_calls[0]["progress_events"] == []
    assert any(
        call["progress_events"] and call["progress_events"][0]["node_name"]
        == "query_handler_node"
        for call in live_state_calls
    )
    assert dummy_st.spinner_calls == []
    assert run_workflow_page._EMPTY_RESULT_PROMPT not in dummy_st.info_calls


def test_active_submitted_run_shows_live_status_not_idle_prompt(monkeypatch) -> None:
    dummy_st = _DummyRenderStreamlit(button_result=False)
    monkeypatch.setattr(run_workflow_page, "st", dummy_st)
    monkeypatch.setattr(
        run_workflow_page,
        "_build_controls",
        lambda: WorkflowControls(
            include_blog=True,
            include_linkedin=False,
            include_research=False,
            include_image=False,
            export_enabled=False,
            export_formats=[],
        ),
    )

    store: Dict[str, Any] = {
        "execution_status": "running",
        "last_error": "",
        "last_result": None,
        "progress_events": [],
        "node_statuses": {},
        "status_messages": ["Workflow started."],
        "last_submission": {"requested_outputs": ["blog"]},
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
        run_workflow_page, "get_progress_events", lambda: list(store["progress_events"])
    )
    monkeypatch.setattr(
        run_workflow_page, "get_node_statuses", lambda: dict(store["node_statuses"])
    )
    monkeypatch.setattr(
        run_workflow_page, "get_status_messages", lambda: list(store["status_messages"])
    )
    monkeypatch.setattr(
        run_workflow_page, "get_last_submission", lambda: dict(store["last_submission"])
    )

    live_state_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        run_workflow_page,
        "_render_active_execution_state",
        lambda **kwargs: live_state_calls.append(
            {
                "execution_status": kwargs["execution_status"],
                "progress_events": list(kwargs["progress_events"]),
                "status_messages": list(kwargs["status_messages"]),
            }
        ),
    )

    run_workflow_page.render()

    assert len(live_state_calls) == 1
    assert live_state_calls[0]["execution_status"] == "running"
    assert live_state_calls[0]["progress_events"] == []
    assert live_state_calls[0]["status_messages"] == ["Workflow started."]
    assert run_workflow_page._EMPTY_RESULT_PROMPT not in dummy_st.info_calls


def test_active_execution_waiting_message_is_scoped_to_node_status(monkeypatch) -> None:
    dummy_st = _DummyRenderStreamlit(button_result=False)
    monkeypatch.setattr(run_workflow_page, "st", dummy_st)
    monkeypatch.setattr(
        run_workflow_page,
        "render_execution_indicators",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        run_workflow_page,
        "render_node_execution_statuses",
        lambda _progress_events: None,
    )

    run_workflow_page._render_active_execution_state(
        container=None,
        execution_status="running",
        progress_events=[],
        status_messages=[],
    )

    assert run_workflow_page._WAITING_FOR_EVENT_MESSAGE in dummy_st.caption_calls
    assert run_workflow_page._EMPTY_RESULT_PROMPT not in dummy_st.info_calls
