from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from frontend.pages import run_workflow as run_workflow_page


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


def test_export_formats_control_remains_interactive_when_export_is_enabled(monkeypatch) -> None:
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
