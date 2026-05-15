# ContentBlitz Testing Strategy

## Overview

ContentBlitz uses layered tests and non-live validators to verify workflow routing, UI rendering, export safety, persistence/restore behavior, reducer merge stability, and provider contract handling.

## Test Layers

### Unit Tests (`tests/unit`)

Validate isolated behavior for:

- tool contracts and error normalization
- cache key generation and TTL behavior
- cost control helpers
- individual agent node state updates
- retry and routing logic
- UI rendering/status/progress helpers
- persistence serialization/session store safety
- export builders and export validators
- guardrails/sanitization helpers

### Integration Tests (`tests/integration`)

Validate multi-agent orchestration and user-facing behavior for:

- graph path correctness
- provider contract regressions (mocked)
- cache read/write integration
- cost control enforcement
- output assembly and export compatibility
- frontend workflow submission and rendering integration
- session save/restore integration
- degraded workflow behavior in UI
- no-real-network behavior by default

### Optional Live Tests (`tests/live`)

Validate real provider connectivity when explicitly enabled.

Rules:

- skip by default
- never required for normal developer validation
- may fail due to provider/network conditions

## Default Validation Path (Non-Live)

Phase 3 validator:

```bash
python scripts/validate_phase3.py --dry-run
```

This script validates, without live provider calls:

1. environment/package readiness for UI/export paths
2. UI imports and frontend wiring
3. export pipeline + non-live export generation
4. dry-run mocked workflow behavior (including degraded/prompt-injection/image-recoverable paths)
5. session serialization/save/load/restore behavior

Phase 2 validator:

```bash
python scripts/validate_phase2.py
```

This script runs:

1. unit/integration suite with coverage
2. live tests in skip-validation mode
3. live smoke script in `--dry-run` mode

## Core Commands

Unit + integration with coverage:

```bash
pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing
```

Verify live tests skip when flags are off:

```bash
pytest tests/live -rs
```

Live smoke audit dry-run:

```bash
python scripts/dev/smoke_phase2_live.py --dry-run
```

## Optional Live Smoke Commands

Text and search:

```bash
CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_generate_text.py -s -rs
CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_search_web.py -s -rs
```

Image:

```bash
CONTENTBLITZ_RUN_LIVE_TESTS=1 CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=1 pytest tests/live/test_live_generate_image.py -s -rs
```

## Deterministic Safety Rules

- unit/integration tests must not require API keys
- unit/integration tests must not call live providers
- provider clients are mocked in contract and failure suites
- live tests are opt-in only
- frontend restore must not rerun providers
- exports are validated before being marked completed

## Coverage

Coverage is enforced on `contentblitz` modules in standard validation flows.

Note:

- on some Windows/OneDrive setups, `.coverage` file locks can occur
- `scripts/validate_phase2.py` uses a temp `COVERAGE_FILE` path to avoid local lock failures
- `scripts/validate_phase3.py` also avoids unsafe temp cleanup assumptions during validation runs
