from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from frontend import app as app_module


@dataclass
class _DummyStreamlit:
    markdown_calls: list[str] = field(default_factory=list)
    info_calls: list[str] = field(default_factory=list)
    warning_calls: list[str] = field(default_factory=list)
    caption_calls: list[str] = field(default_factory=list)
    session_state: dict[str, Any] = field(default_factory=dict)

    def markdown(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.markdown_calls.append(str(value))

    def info(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.info_calls.append(str(value))

    def warning(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.warning_calls.append(str(value))

    def caption(self, value: str, *_args: Any, **_kwargs: Any) -> None:
        self.caption_calls.append(str(value))


@pytest.mark.parametrize(
    ("status", "status_label", "expected_info_count", "expected_warning_count"),
    [
        ("disabled", "Disabled", 1, 0),
        ("enabled", "Enabled", 0, 0),
        ("degraded", "Degraded", 0, 1),
    ],
)
def test_ui_observability_status_renders_safely(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    status_label: str,
    expected_info_count: int,
    expected_warning_count: int,
) -> None:
    dummy_st = _DummyStreamlit(session_state={"keep": "unchanged"})
    monkeypatch.setattr(app_module, "st", dummy_st)
    monkeypatch.setattr(
        app_module,
        "build_observability_diagnostics",
        lambda: {
            "status": status,
            "status_label": status_label,
            "status_tone_class": "cbx-status-green",
            "tracing_enabled": status == "enabled",
            "project_name": "ContentBlitz",
            "endpoint_host": "api.smith.langchain.com",
            "last_trace_attempt_label": "Ready",
            "note": "Safe diagnostics note.",
            "dashboard_instruction": (
                "For trace details, review the LangSmith dashboard manually."
            ),
        },
    )

    before_state = dict(dummy_st.session_state)
    app_module._render_observability_status()  # noqa: SLF001

    rendered = "\n".join(dummy_st.markdown_calls + dummy_st.caption_calls).lower()
    assert status_label.lower() in rendered
    assert "contentblitz" in rendered
    assert "api.smith.langchain.com" in rendered
    assert "langsmith_api_key" not in rendered
    assert "openai_api_key" not in rendered
    assert "traceback" not in rendered
    assert len(dummy_st.info_calls) == expected_info_count
    assert len(dummy_st.warning_calls) == expected_warning_count
    assert dict(dummy_st.session_state) == before_state


def test_ui_observability_fallback_is_safe_when_diagnostics_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(app_module, "st", dummy_st)
    monkeypatch.setattr(
        app_module,
        "build_observability_diagnostics",
        lambda: (_ for _ in ()).throw(RuntimeError("unexpected traceback")),
    )

    app_module._render_observability_status()  # noqa: SLF001

    rendered = "\n".join(
        dummy_st.markdown_calls
        + dummy_st.info_calls
        + dummy_st.warning_calls
        + dummy_st.caption_calls
    ).lower()
    assert "degraded" in rendered
    assert "temporarily unavailable" in rendered
    assert "traceback" not in rendered


def test_ui_observability_render_does_not_call_langsmith_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contentblitz.core import observability as observability_module

    dummy_st = _DummyStreamlit()
    monkeypatch.setattr(app_module, "st", dummy_st)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    def _unexpected_import(_name: str) -> object:
        raise AssertionError("Direct LangSmith import is not expected in UI render.")

    monkeypatch.setattr(observability_module, "import_module", _unexpected_import)
    app_module._render_observability_status()  # noqa: SLF001

    assert dummy_st.markdown_calls
