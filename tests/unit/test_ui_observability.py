from __future__ import annotations

import json

import pytest

from contentblitz.core.observability import ObservabilityConfig
from contentblitz.ui.observability import build_observability_diagnostics


def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_ui_observability_renders_disabled_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    diagnostics = build_observability_diagnostics()

    assert diagnostics["status"] == "disabled"
    assert diagnostics["status_label"] == "Disabled"
    assert diagnostics["tracing_enabled"] is False
    assert diagnostics["last_trace_attempt_status"] == "not_requested"
    assert diagnostics["endpoint_host"]


def test_ui_observability_renders_enabled_status_with_mocked_config() -> None:
    diagnostics = build_observability_diagnostics(
        config=ObservabilityConfig(
            tracing_requested=True,
            tracing_enabled=True,
            trace_sample_rate=1.0,
            trace_failure_sample_rate=1.0,
            endpoint="https://api.smith.langchain.com/v1/runs?debug=true",
            project="ContentBlitz",
            status="enabled",
            message="safe",
        )
    )

    assert diagnostics["status"] == "enabled"
    assert diagnostics["status_label"] == "Enabled"
    assert diagnostics["tracing_enabled"] is True
    assert diagnostics["endpoint_host"] == "api.smith.langchain.com"
    assert diagnostics["project_name"] == "ContentBlitz"


def test_ui_observability_renders_degraded_status_safely() -> None:
    diagnostics = build_observability_diagnostics(
        config=ObservabilityConfig(
            tracing_requested=True,
            tracing_enabled=False,
            trace_sample_rate=1.0,
            trace_failure_sample_rate=1.0,
            endpoint="https://api.smith.langchain.com",
            project="ContentBlitz",
            status="degraded",
            message=(
                "Traceback (most recent call last):\n"
                '  File "bad.py", line 1\n'
                "RuntimeError: bad"
            ),
        )
    )

    encoded = json.dumps(diagnostics, sort_keys=True).lower()
    assert diagnostics["status"] == "degraded"
    assert diagnostics["status_label"] == "Degraded"
    assert diagnostics["last_trace_attempt_status"] == "unavailable"
    assert "traceback" not in encoded
    assert "runtimeerror" not in encoded


def test_ui_observability_hides_api_keys_and_env_key_names() -> None:
    diagnostics = build_observability_diagnostics(
        config=ObservabilityConfig(
            tracing_requested=True,
            tracing_enabled=True,
            trace_sample_rate=1.0,
            trace_failure_sample_rate=1.0,
            endpoint="https://api.smith.langchain.com",
            project="ContentBlitz",
            status="enabled",
            message=(
                "LANGSMITH_API_KEY=ls-secret OPENAI_API_KEY=sk-secret "
                "LANGSMITH_ENDPOINT=https://api.smith.langchain.com"
            ),
        )
    )
    encoded = json.dumps(diagnostics, sort_keys=True)

    assert "ls-secret" not in encoded
    assert "sk-secret" not in encoded
    assert "LANGSMITH_API_KEY" not in encoded
    assert "OPENAI_API_KEY" not in encoded
    assert "LANGSMITH_ENDPOINT" not in encoded


def test_ui_observability_does_not_import_langsmith(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contentblitz.core import observability as observability_module

    _clear_langsmith_env(monkeypatch)

    def _unexpected_import(_name: str) -> object:
        raise AssertionError("LangSmith should not be imported for UI diagnostics.")

    monkeypatch.setattr(observability_module, "import_module", _unexpected_import)
    diagnostics = build_observability_diagnostics()

    assert diagnostics["status"] == "disabled"
