# Phase 2 Live Smoke Tests

## Purpose
These optional live smoke tests validate real provider connectivity and normalized response behavior for Phase 2 tools:
- `generate_text` (OpenAI)
- `search_web` (SERP and `auto`)
- `generate_image` (DALL-E)

They are a lightweight production-sanity layer in addition to mocked unit/integration tests.

## Validation Script vs Live Smoke
- `scripts/validate_phase2.py`:
  - safe, non-live validation entrypoint
  - does not require API keys
  - does not execute real provider calls
  - validates that live tests skip safely by default
  - runs the live smoke script in `--dry-run` mode only
- `scripts/dev/smoke_phase2_live.py`:
  - optional live smoke/audit utility
  - intended for real provider connectivity checks when flags are enabled
  - may incur provider usage cost when run in live mode

## Why Live Tests Are Skipped By Default
Live provider tests are opt-in because they:
- require API keys
- can incur usage cost
- depend on external network/provider availability

Default behavior:
- live pytest tests are skipped unless `CONTENTBLITZ_RUN_LIVE_TESTS=1`
- live image tests also require `CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=1`

## Required Environment Variables
Set these in local `.env` when you want to run live checks:
- `OPENAI_API_KEY`
- `SERP_API_KEY`
- `PERPLEXITY_API_KEY` (needed for `provider="auto"` fallback scenarios)
- `CONTENTBLITZ_RUN_LIVE_TESTS=1`
- `CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=1` (image smoke only)

Never commit `.env`.

## Cost Warning
Live smoke tests call real providers and may consume paid quota. Run only when needed.

## Live Test Locations
- `tests/live/test_live_generate_text.py`
- `tests/live/test_live_search_web.py`
- `tests/live/test_live_generate_image.py`
- `tests/live/test_live_openai_agent_integration.py`

## Manual Script Locations
- `scripts/dev/manual_live_openai_agent.py`
- `scripts/dev/manual_live_web_search.py`
- `scripts/dev/manual_cache_check.py`
- `scripts/dev/smoke_phase2_live.py`

## Run Commands
Normal validation (no live APIs):

```bash
pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing
```

Confirm live tests skip safely when flags are off:

```bash
pytest tests/live -rs
```

Dry-run audit (no provider calls):

```bash
python scripts/dev/smoke_phase2_live.py --dry-run
```

Optional live text/search:

```bash
CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_generate_text.py -s -rs
CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_search_web.py -s -rs
```

Optional live image:

```bash
CONTENTBLITZ_RUN_LIVE_TESTS=1 CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=1 pytest tests/live/test_live_generate_image.py -s -rs
```

## Mocked Tests vs Live Tests
- `tests/unit` + `tests/integration`:
  - use mocks/stubs
  - never require real API keys
  - should never make real network calls
- `tests/live`:
  - optional, real provider calls
  - explicitly gated by environment flags

## Troubleshooting Skipped Tests
- If tests show skipped:
  - verify `CONTENTBLITZ_RUN_LIVE_TESTS=1`
  - for image, also verify `CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=1`
  - verify provider key environment variables are present
- use `pytest tests/live -rs` to view skip reasons
