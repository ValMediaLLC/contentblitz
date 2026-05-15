# Validation Framework

## Scope

This document describes non-live validation entrypoints and what they verify.

## Phase 3 Validator

Primary script:

- `scripts/validate_phase3.py`

Commands:

```bash
python scripts/validate_phase3.py
python scripts/validate_phase3.py --dry-run
```

Both modes are non-live and deterministic.

## Phase 3 Checks

`validate_phase3.py` validates:

- environment readiness
  - Python version compatibility
  - required imports (including Streamlit)
  - writable export/session/tmp locations
  - no provider key requirement
- UI readiness
  - frontend import graph
  - run/history page imports
  - rendering/status imports
- export readiness
  - markdown/html/pdf/docx builders and validators
  - non-live export generation path
- dry-run workflow behavior
  - blog path
  - degraded research-safe path
  - multi-output aggregation path
  - prompt-injection-safe path
  - recoverable image failure path
  - deterministic export metadata check
- persistence/restore readiness
  - safe serialization
  - save/load/deserialize
  - restore render payload generation
  - no workflow/provider rerun during restore

## Output and Exit Codes

- Status lines use the same symbol pattern as other validators:
  - `✅` / `[OK]`
  - `⚠️` / `[WARN]`
  - `❌` / `[FAIL]`
- Exit codes:
  - `0` = success
  - non-zero = validation failure

## Provider/Network Behavior

- No live provider calls are required.
- API keys are optional for validator success.
- Network access is blocked in validator dry-run workflow checks.

## Related Validators

- `scripts/validate_phase1.py` for structural baseline checks.
- `scripts/validate_phase2.py` for Phase 2 non-live provider/cost/cache/live-skip gating checks.

## Test Coverage for Validator

Validator-specific tests:

- `tests/unit/test_validate_phase3.py`
- `tests/integration/test_phase3_validator.py`
