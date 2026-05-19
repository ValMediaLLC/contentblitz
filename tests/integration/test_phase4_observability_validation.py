from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(relative_path: str, module_name: str) -> Any:
    script_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module at {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_phase4_runs_expected_checks() -> None:
    module = _load_script_module(
        "scripts/validate_phase4.py",
        "validate_phase4_script_test",
    )
    pytest_calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_pytest_runner(name: str, targets: tuple[str, ...]):
        pytest_calls.append((name, tuple(targets)))
        return module.CheckResult(name, True, "mocked")

    results = module.run_validation(
        dry_run=False,
        verbose=False,
        pytest_runner=_fake_pytest_runner,
    )
    result_names = [result.name for result in results]

    assert "Observability config imports cleanly" in result_names
    assert "Tracing disabled path works" in result_names
    assert "Redaction helpers pass" in result_names
    assert "Graph tracing tests pass" in result_names
    assert "UI observability tests pass" in result_names
    assert "Unit/integration checks run without LangSmith credentials" in result_names
    assert pytest_calls
    assert all(result.passed for result in results)


def test_validate_phase4_main_dry_run_reports_expected_summary(capsys) -> None:
    module = _load_script_module(
        "scripts/validate_phase4.py",
        "validate_phase4_script_dry_run_test",
    )
    exit_code = module.main(["--dry-run"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Dry-run mode enabled" in output
    assert "Observability config imports cleanly" in output
    assert "UI observability tests pass" in output


def test_smoke_langsmith_dry_run_makes_no_live_trace_attempt(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script_module(
        "scripts/dev/smoke_langsmith.py",
        "smoke_langsmith_script_dry_run_test",
    )
    monkeypatch.setattr(module, "_maybe_load_dotenv", lambda: None)
    monkeypatch.delenv("CONTENTBLITZ_RUN_LANGSMITH_SMOKE", raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-secret-test-value")
    monkeypatch.setattr(
        module,
        "_attempt_live_trace",
        lambda: (_ for _ in ()).throw(
            AssertionError("Live trace should not run in dry-run mode.")
        ),
    )

    exit_code = module.main(["--dry-run"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "No LangSmith calls were made." in output
    assert "ls-secret-test-value" not in output


def test_smoke_langsmith_live_is_skipped_unless_flag_is_set(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script_module(
        "scripts/dev/smoke_langsmith.py",
        "smoke_langsmith_script_gate_test",
    )
    monkeypatch.setattr(module, "_maybe_load_dotenv", lambda: None)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.delenv("CONTENTBLITZ_RUN_LANGSMITH_SMOKE", raising=False)
    monkeypatch.setattr(
        module,
        "_attempt_live_trace",
        lambda: (_ for _ in ()).throw(
            AssertionError("Live trace should not run without opt-in flag.")
        ),
    )

    exit_code = module.main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Live smoke skipped." in output


def test_smoke_langsmith_missing_key_is_handled_safely(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script_module(
        "scripts/dev/smoke_langsmith.py",
        "smoke_langsmith_script_missing_key_test",
    )
    monkeypatch.setattr(module, "_maybe_load_dotenv", lambda: None)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("CONTENTBLITZ_RUN_LANGSMITH_SMOKE", "1")
    monkeypatch.setattr(
        module,
        "_attempt_live_trace",
        lambda: (_ for _ in ()).throw(
            AssertionError("Live trace should not run when tracing is unavailable.")
        ),
    )

    exit_code = module.main([])
    output = capsys.readouterr().out.lower()

    assert exit_code == 0
    assert "not attempted" in output
    assert "traceback" not in output
    assert "langsmith_api_key=" not in output


def test_smoke_langsmith_output_does_not_expose_secrets_or_stack_trace(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script_module(
        "scripts/dev/smoke_langsmith.py",
        "smoke_langsmith_script_output_safety_test",
    )
    monkeypatch.setattr(module, "_maybe_load_dotenv", lambda: None)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-ultra-secret-value")
    monkeypatch.delenv("CONTENTBLITZ_RUN_LANGSMITH_SMOKE", raising=False)

    exit_code = module.main(["--dry-run"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "ls-ultra-secret-value" not in output
    assert "traceback (most recent call last)" not in output.lower()
    assert "api key" not in output.lower() or "present: true" in output.lower()


def test_smoke_langsmith_live_attempt_runs_when_explicitly_enabled(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script_module(
        "scripts/dev/smoke_langsmith.py",
        "smoke_langsmith_script_live_enabled_test",
    )
    monkeypatch.setattr(module, "_maybe_load_dotenv", lambda: None)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_RUN_LANGSMITH_SMOKE", "1")
    monkeypatch.setattr(
        module,
        "build_observability_diagnostics",
        lambda: {
            "status_label": "Enabled",
            "tracing_enabled": True,
            "project_name": "ContentBlitz",
            "endpoint_host": "api.smith.langchain.com",
            "last_trace_attempt_label": "Ready",
        },
    )
    monkeypatch.setattr(module, "_attempt_live_trace", lambda: (True, "safe summary"))

    exit_code = module.main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Live smoke result: success" in output
    assert "safe summary" in output
