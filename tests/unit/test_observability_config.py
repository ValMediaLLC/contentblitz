from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from contentblitz.config import (
    LANGSMITH_ENDPOINT_DEFAULT,
    LANGSMITH_PROJECT_DEFAULT,
)
from contentblitz.core.observability import (
    build_observability_config,
    is_tracing_enabled,
    observability_summary,
)


def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_tracing_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_langsmith_env(monkeypatch)

    config = build_observability_config()

    assert config.tracing_requested is False
    assert config.tracing_enabled is False
    assert config.endpoint == LANGSMITH_ENDPOINT_DEFAULT
    assert config.project == LANGSMITH_PROJECT_DEFAULT
    assert config.status == "disabled"
    assert is_tracing_enabled() is False


@pytest.mark.parametrize("truthy_value", ["1", "true", "TRUE", "yes", "on"])
def test_tracing_enabled_only_when_langsmith_tracing_is_truthy(
    monkeypatch: pytest.MonkeyPatch,
    truthy_value: str,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", truthy_value)
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-test")

    config = build_observability_config()

    assert config.tracing_requested is True
    assert config.tracing_enabled is True
    assert config.status == "enabled"

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    config_false = build_observability_config()
    assert config_false.tracing_enabled is False


def test_missing_api_key_degrades_tracing_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    config = build_observability_config()

    assert config.tracing_requested is True
    assert config.tracing_enabled is False
    assert config.status == "degraded"
    assert "missing" in config.message.lower()


def test_public_observability_output_never_exposes_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "sk-super-secret")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://example.langsmith.local")
    monkeypatch.setenv("LANGSMITH_PROJECT", "cbx-observability")

    summary = observability_summary()

    assert "api_key" not in summary
    assert "LANGSMITH_API_KEY" not in summary
    assert "sk-super-secret" not in repr(summary)
    assert summary["endpoint"] == "https://example.langsmith.local"
    assert summary["project"] == "cbx-observability"


def test_config_import_does_not_call_langsmith(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_langsmith_modules: list[str] = []
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if str(name).startswith("langsmith"):
            imported_langsmith_modules.append(str(name))
            raise AssertionError("LangSmith import should not happen at import time.")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as context:
        context.setattr(builtins, "__import__", _guarded_import)
        sys.modules.pop("contentblitz.core.observability", None)
        sys.modules.pop("contentblitz.config", None)
        importlib.import_module("contentblitz.config")
        importlib.import_module("contentblitz.core.observability")

    assert imported_langsmith_modules == []


def test_observability_helpers_work_without_langsmith_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)

    config = build_observability_config()
    summary = observability_summary()

    assert config.tracing_enabled is False
    assert summary["tracing_enabled"] is False
    assert summary["project"] == LANGSMITH_PROJECT_DEFAULT
