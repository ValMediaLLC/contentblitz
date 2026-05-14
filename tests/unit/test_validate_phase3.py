from __future__ import annotations

from pathlib import Path

import scripts.validate_phase3 as validate_phase3


def _stub_all_checks(monkeypatch, validator: validate_phase3.Phase3Validator) -> None:
    monkeypatch.setattr(validator, "validate_environment", lambda: None)
    monkeypatch.setattr(validator, "validate_ui_imports", lambda: None)
    monkeypatch.setattr(validator, "validate_export_pipeline", lambda: None)
    monkeypatch.setattr(validator, "validate_non_live_export_generation", lambda: None)
    monkeypatch.setattr(validator, "validate_workflow_dry_run", lambda: None)
    monkeypatch.setattr(validator, "validate_session_restore", lambda: None)


def test_validate_phase3_main_returns_validator_status(monkeypatch) -> None:
    monkeypatch.setattr(validate_phase3.Phase3Validator, "run", lambda self: 0)
    assert validate_phase3.main(["--dry-run"]) == 0


def test_validate_phase3_force_fail_returns_non_zero(monkeypatch) -> None:
    validator = validate_phase3.Phase3Validator(dry_run=True, verbose=False)
    _stub_all_checks(monkeypatch, validator)
    monkeypatch.setenv("CONTENTBLITZ_PHASE3_FORCE_FAIL", "1")

    assert validator.run() == 1


def test_validate_phase3_environment_validation_succeeds_without_provider_keys(
    monkeypatch, tmp_path: Path
) -> None:
    validator = validate_phase3.Phase3Validator(dry_run=True, verbose=False)

    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    monkeypatch.setenv("CONTENTBLITZ_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    validator.validate_environment()

