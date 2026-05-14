from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_phase3.py"


def _assert_ok_status_line(output: str, label: str) -> None:
    candidates = (
        f"✅ {label}",
        f"[OK] {label}",
    )
    assert any(candidate in output for candidate in candidates), output


def _base_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CONTENTBLITZ_EXPORT_DIR"] = str(tmp_path / "exports")
    env["CONTENTBLITZ_SESSION_DIR"] = str(tmp_path / "sessions")
    env.pop("OPENAI_API_KEY", None)
    env.pop("SERP_API_KEY", None)
    env.pop("PERPLEXITY_API_KEY", None)
    env.pop("CONTENTBLITZ_PHASE3_FORCE_FAIL", None)
    return env


def test_validate_phase3_dry_run_succeeds_without_provider_keys(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        env=_base_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode == 0, combined
    _assert_ok_status_line(combined, "UI imports")
    _assert_ok_status_line(combined, "Export validation")
    _assert_ok_status_line(combined, "Session restore validation")
    _assert_ok_status_line(combined, "Dry-run workflow validation")
    _assert_ok_status_line(combined, "Non-live export generation")


def test_validate_phase3_returns_non_zero_on_forced_failure(tmp_path: Path) -> None:
    env = _base_env(tmp_path)
    env["CONTENTBLITZ_PHASE3_FORCE_FAIL"] = "1"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined = f"{completed.stdout}\n{completed.stderr}"
    assert completed.returncode != 0, combined
    assert "FINAL RESULT: PHASE 3 VALIDATION FAILED" in combined
