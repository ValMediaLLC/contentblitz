#!/usr/bin/env python3
"""Safe Phase 4 observability validation runner (non-live by default)."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LANGSMITH_ENV_KEYS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
)

PYTEST_REDACTION = ("tests/unit/test_observability_redaction.py",)
PYTEST_GRAPH_TRACING = ("tests/integration/test_observability_graph_tracing.py",)
PYTEST_UI_OBSERVABILITY = (
    "tests/unit/test_ui_observability.py",
    "tests/integration/test_observability_ui_status.py",
)
PYTEST_NO_CREDENTIALS = (
    "tests/unit/test_observability_config.py::"
    "test_observability_helpers_work_without_langsmith_credentials",
    "tests/integration/test_observability_graph_tracing.py::"
    "test_graph_executes_with_tracing_disabled",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _symbol(emoji: str, fallback: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        emoji.encode(encoding)
        return emoji
    except Exception:
        return fallback


OK_SYMBOL = _symbol("✅", "[OK]")
FAIL_SYMBOL = _symbol("❌", "[FAIL]")
WARN_SYMBOL = _symbol("⚠️", "[WARN]")


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _format_result(result: CheckResult) -> None:
    suffix = f" ({result.detail})" if result.detail else ""
    if result.passed:
        print(f"{OK_SYMBOL} {result.name}{suffix}")
        return
    print(f"{FAIL_SYMBOL} {result.name}{suffix}")


def _without_langsmith_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in LANGSMITH_ENV_KEYS:
        env.pop(key, None)
    return env


def check_observability_imports() -> CheckResult:
    try:
        config_module = importlib.import_module("contentblitz.config")
        observability_module = importlib.import_module(
            "contentblitz.core.observability"
        )
    except Exception as exc:  # pragma: no cover - defensive script failure guard
        return CheckResult("Observability config imports cleanly", False, str(exc))

    required = (
        "LANGSMITH_ENDPOINT_DEFAULT",
        "LANGSMITH_PROJECT_DEFAULT",
        "build_observability_config",
        "safe_trace_metadata",
    )
    missing = [
        item
        for item in required
        if not hasattr(config_module, item) and not hasattr(observability_module, item)
    ]
    if missing:
        return CheckResult(
            "Observability config imports cleanly",
            False,
            f"missing symbols: {', '.join(missing)}",
        )
    return CheckResult("Observability config imports cleanly", True)


def check_tracing_disabled_path() -> CheckResult:
    try:
        observability_module = importlib.import_module(
            "contentblitz.core.observability"
        )
        saved = {key: os.environ.get(key) for key in LANGSMITH_ENV_KEYS}
        try:
            for key in LANGSMITH_ENV_KEYS:
                os.environ.pop(key, None)
            config = observability_module.build_observability_config()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    except Exception as exc:  # pragma: no cover - defensive script failure guard
        return CheckResult("Tracing disabled path works", False, str(exc))

    if config.tracing_enabled:
        return CheckResult(
            "Tracing disabled path works",
            False,
            "tracing_enabled should be false when LangSmith env vars are absent",
        )
    if str(config.status).strip().lower() != "disabled":
        return CheckResult(
            "Tracing disabled path works",
            False,
            f"unexpected status: {config.status}",
        )
    return CheckResult("Tracing disabled path works", True)


def _run_pytest_check(
    name: str,
    targets: Sequence[str],
    *,
    verbose: bool = False,
) -> CheckResult:
    command = [sys.executable, "-m", "pytest", *targets]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=str(ROOT),
        env=_without_langsmith_env(),
        capture_output=not verbose,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return CheckResult(name, True)

    detail = f"exit code {completed.returncode}"
    if not verbose:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        if stderr:
            detail = f"{detail}; stderr={stderr.splitlines()[-1][:120]}"
        elif stdout:
            detail = f"{detail}; stdout={stdout.splitlines()[-1][:120]}"
    return CheckResult(name, False, detail)


def run_validation(
    *,
    dry_run: bool = False,
    verbose: bool = False,
    pytest_runner: Callable[[str, Sequence[str]], CheckResult] | None = None,
) -> list[CheckResult]:
    runner = pytest_runner or (
        lambda name, targets: _run_pytest_check(name, targets, verbose=verbose)
    )
    checks: list[CheckResult] = [
        check_observability_imports(),
        check_tracing_disabled_path(),
    ]
    if dry_run:
        checks.extend(
            [
                CheckResult("Redaction helpers pass", True, "dry-run"),
                CheckResult("Graph tracing tests pass", True, "dry-run"),
                CheckResult("UI observability tests pass", True, "dry-run"),
                CheckResult(
                    "Unit/integration checks run without LangSmith credentials",
                    True,
                    "dry-run",
                ),
            ]
        )
        return checks

    checks.append(runner("Redaction helpers pass", PYTEST_REDACTION))
    checks.append(runner("Graph tracing tests pass", PYTEST_GRAPH_TRACING))
    checks.append(runner("UI observability tests pass", PYTEST_UI_OBSERVABILITY))
    checks.append(
        runner(
            "Unit/integration checks run without LangSmith credentials",
            PYTEST_NO_CREDENTIALS,
        )
    )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Phase 4 observability safety without live tracing."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show validation checklist without running pytest checks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream pytest output directly for failing checks.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _print_header("ContentBlitz Phase 4 Observability Validation")
    if args.dry_run:
        print(f"{WARN_SYMBOL} Dry-run mode enabled (no pytest checks executed).")

    results = run_validation(
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    for result in results:
        _format_result(result)

    failed = [result for result in results if not result.passed]
    _print_header("Validation Summary")
    print(f"Checks run: {len(results)}")
    print(f"Passed: {len(results) - len(failed)}")
    print(f"Failed: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
