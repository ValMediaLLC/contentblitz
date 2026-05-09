# ContentBlitz Testing Strategy

## Overview

ContentBlitz uses layered testing to validate routing, contracts, state ownership, provider failure handling, cache behavior, and cost-control safety.

## Test Layers

### Unit Tests (`tests/unit`)

Validate isolated behavior for:

- tool contracts and error normalization
- cache key generation and TTL behavior
- cost control helpers
- individual agent node state updates
- retry and routing logic

### Integration Tests (`tests/integration`)

Validate multi-agent orchestration and end-to-end behavior for:

- graph path correctness
- provider contract regressions (mocked)
- cache read/write integration
- cost control enforcement
- output assembly and export compatibility
- no-real-network behavior by default

### Optional Live Tests (`tests/live`)

Validate real provider connectivity when explicitly enabled.

Rules:

- skip by default
- never required for normal developer validation
- may fail due to provider/network conditions

## Default Validation Path (Non-Live)

Recommended command:

```bash
python scripts/validate_phase2.py
```

This script runs:

1. unit/integration suite with coverage
2. live tests in skip-validation mode
3. live smoke script in `--dry-run` mode

## Direct Commands

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

## Coverage

Coverage is enforced on `contentblitz` modules in standard validation flows.

Note:

- on some Windows/OneDrive setups, `.coverage` file locks can occur
- `scripts/validate_phase2.py` uses a temp `COVERAGE_FILE` path to avoid local lock failures
