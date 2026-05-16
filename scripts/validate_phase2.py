#!/usr/bin/env python3
"""Safe Phase 2 validation entrypoint (non-live)."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def _symbol(emoji: str, fallback: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        emoji.encode(encoding)
        return emoji
    except Exception:
        return fallback


OK_SYMBOL = _symbol("✅", "[OK]")
WARN_SYMBOL = _symbol("⚠️", "[WARN]")
FAIL_SYMBOL = _symbol("❌", "[FAIL]")


def _ok(message: str) -> None:
    print(f"{OK_SYMBOL} {message}")


def _warn(message: str) -> None:
    print(f"{WARN_SYMBOL} {message}")


def _fail(message: str) -> None:
    print(f"{FAIL_SYMBOL} {message}")


def _print_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _stream_pipe(pipe, sink, collector: list[str]) -> None:
    for line in iter(pipe.readline, ""):
        collector.append(line)
        sink.write(line)
        sink.flush()
    pipe.close()


def _run_command(
    *,
    name: str,
    command: list[str],
    env_overrides: dict[str, str] | None = None,
    verbose: bool = False,
) -> CommandResult:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    if verbose:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        out_lines: list[str] = []
        err_lines: list[str] = []
        out_thread = threading.Thread(
            target=_stream_pipe,
            args=(process.stdout, sys.stdout, out_lines),
            daemon=True,
        )
        err_thread = threading.Thread(
            target=_stream_pipe,
            args=(process.stderr, sys.stderr, err_lines),
            daemon=True,
        )
        out_thread.start()
        err_thread.start()
        returncode = process.wait()
        out_thread.join()
        err_thread.join()
        stdout = "".join(out_lines)
        stderr = "".join(err_lines)
    else:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

    return CommandResult(
        name=name,
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _extract_pytest_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in ("passed", "failed", "skipped", "error", "errors", "xfailed", "xpassed"):
        match = re.search(rf"(\d+)\s+{key}\b", output, flags=re.IGNORECASE)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _extract_warning_count(output: str) -> int:
    match = re.search(r"(\d+)\s+warning[s]?\b", output, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def _extract_coverage_percent(output: str) -> int | None:
    match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
    if not match:
        return None
    return int(match.group(1))


def _pytest_high_level_summary(output: str) -> str:
    counts = _extract_pytest_counts(output)
    parts: list[str] = []

    passed = counts.get("passed")
    skipped = counts.get("skipped")
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    if passed is not None:
        parts.append(f"{passed} passed")
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed/errors")

    warnings = _extract_warning_count(output)
    if warnings:
        parts.append(f"{warnings} warning")

    return ", ".join(parts) if parts else "pytest completed"


def _print_failure_debug(result: CommandResult) -> None:
    print(f"Command: {' '.join(result.command)}")
    print(f"Exit code: {result.returncode}")
    if result.stdout.strip():
        print("\nSTDOUT:")
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("\nSTDERR:")
        print(result.stderr.rstrip())


def _validate_live_skip_behavior(result: CommandResult) -> tuple[bool, str]:
    combined = f"{result.stdout}\n{result.stderr}"
    counts = _extract_pytest_counts(combined)
    skipped = counts.get("skipped", 0)
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)

    if result.returncode != 0:
        return False, "Live test command returned non-zero exit."
    if failed > 0:
        return False, "Live tests reported failures/errors."
    if skipped <= 0:
        return (
            False,
            "Live tests were expected to skip by default but no skips were reported.",
        )
    if passed > 0:
        return (
            False,
            "Live tests were expected to skip by default but some tests passed.",
        )
    return True, f"Skipped tests: {skipped}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ContentBlitz Phase 2 readiness without live provider calls."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream subprocess output live instead of condensed output.",
    )
    parser.add_argument(
        "--skip-coverage",
        action="store_true",
        help="Run unit/integration suite without coverage flags for faster local iteration.",
    )
    args = parser.parse_args()

    print("\nContentBlitz Phase 2 Validation\n" + "-" * 33)
    _ok("Mode: safe/non-live")

    phase2_results: list[tuple[str, bool]] = []
    had_failure = False

    pytest_base = [sys.executable, "-m", "pytest", "tests/unit", "tests/integration"]
    if args.skip_coverage:
        suite_command = list(pytest_base)
        suite_env = None
    else:
        suite_command = [
            *pytest_base,
            "--cov=contentblitz",
            "--cov-report=term-missing",
        ]
        coverage_file = (
            Path(tempfile.gettempdir())
            / f"contentblitz_phase2_validate_{os.getpid()}_{uuid.uuid4().hex}.coverage"
        )
        suite_env = {"COVERAGE_FILE": str(coverage_file)}

    _print_header("Unit/Integration Suite")
    suite_result = _run_command(
        name="Unit/integration suite",
        command=suite_command,
        env_overrides=suite_env,
        verbose=args.verbose,
    )
    suite_passed = suite_result.returncode == 0
    if suite_passed:
        combined = f"{suite_result.stdout}\n{suite_result.stderr}"
        summary = _pytest_high_level_summary(combined)
        coverage_percent = _extract_coverage_percent(combined)
        if not args.skip_coverage and coverage_percent is not None:
            _ok(
                f"Unit/integration suite passed ({summary}; coverage {coverage_percent}%)"
            )
        else:
            _ok(f"Unit/integration suite passed ({summary})")
    else:
        _fail("Unit/integration suite failed")
        _print_failure_debug(suite_result)

    phase2_results.append(("Unit/integration suite", suite_passed))
    had_failure = had_failure or (not suite_passed)

    _print_header("Live Tests Skip Check")
    live_result = _run_command(
        name="Live tests safely skipped",
        command=[sys.executable, "-m", "pytest", "tests/live", "-rs"],
        env_overrides={
            "CONTENTBLITZ_RUN_LIVE_TESTS": "0",
            "CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS": "0",
        },
        verbose=args.verbose,
    )
    live_passed, live_reason = _validate_live_skip_behavior(live_result)
    if live_passed:
        _ok(f"Live tests safely skipped ({live_reason})")
    else:
        _fail("Live tests skip validation failed")
        _warn(live_reason)
        _print_failure_debug(live_result)

    phase2_results.append(("Live tests safely skipped", live_passed))
    had_failure = had_failure or (not live_passed)

    _print_header("Dry-Run Smoke Validation")
    dry_run_result = _run_command(
        name="Dry-run smoke validation",
        command=[sys.executable, "scripts/dev/smoke_phase2_live.py", "--dry-run"],
        verbose=args.verbose,
    )
    dry_passed = dry_run_result.returncode == 0
    if dry_passed:
        _ok("Dry-run smoke validation passed")
    else:
        _fail("Dry-run smoke validation failed")
        _print_failure_debug(dry_run_result)

    phase2_results.append(("Dry-run smoke validation", dry_passed))
    had_failure = had_failure or (not dry_passed)

    non_live_passed = live_passed and dry_passed
    phase2_results.append(("No live provider execution required", non_live_passed))
    if non_live_passed:
        _ok("No live provider execution required")
    else:
        _fail("No live provider execution requirement was not satisfied")

    _print_header("Phase 2 Validation Summary")
    for label, passed in phase2_results:
        if passed:
            _ok(label)
        else:
            _fail(label)

    if had_failure:
        print("")
        _fail("Phase 2 validation failed.")
        print("")
        return 1

    print("")
    _ok("Phase 2 validation passed.")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
