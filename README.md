# ContentBlitz

ContentBlitz is a LangGraph-based multi-agent content orchestration system with Phase 2 provider integrations for text, web research, and image generation.

## Current State

- 12-node workflow is implemented and active.
- Provider-backed tools are implemented for:
  - OpenAI text generation
  - SERP web search with Perplexity fallback support
  - OpenAI DALL-E image generation with model fallback
- Research caching and cost controls are integrated.
- Unit and integration tests are deterministic and non-live by default.

## Phase 2 Integrations

- `contentblitz/tools/generate_text.py`
  - Primary: `gpt-4o`
  - Fallback: `gpt-4o-mini`
  - Retries: `RETRY_POLICY[agent_key] + 1` attempts per model
- `contentblitz/tools/search_web.py`
  - Providers: `serp`, `perplexity`, `auto`
  - `auto` tries SERP first, then Perplexity when SERP is degraded/unusable
  - Normalized result schema via `SearchWebResult`
- `contentblitz/tools/generate_image.py`
  - Primary: `dall-e-3`
  - Fallback: `dall-e-2`
  - Returns URL or provider file reference only (no base64 payloads)

## Cache and Cost Controls

- Research cache key format:
  - `research:{sha256_normalized_query}:{depth}`
- Default cache backend:
  - in-memory, process-local
- Default cache TTL:
  - 1800 seconds
- Optional persistent cache prototype:
  - Set `CONTENTBLITZ_CACHE_BACKEND=sqlite` to enable local SQLite cache
  - Set `CONTENTBLITZ_CACHE_SQLITE_PATH` to choose the local DB path
  - If cache backend env vars are not set, `in_memory` remains the default
- Cost control counters:
  - `tokens_used_this_session`
  - `search_queries_used_this_session`
  - `image_generations_used_this_session`
  - `total_retries_used_this_session`
  - `budget_exceeded`

## Security Baseline

- `.env` is never committed.
- API keys are read only from environment variables.
- Tools are stateless.
- State never stores secrets.
- Provider errors are normalized.
- Base64 image data is never stored in state.

## Setup

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create `.env` for optional live checks:

```env
OPENAI_API_KEY=
SERP_API_KEY=
PERPLEXITY_API_KEY=
CONTENTBLITZ_RUN_LIVE_TESTS=0
CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=0

# Optional cache backend configuration
# If unset, ContentBlitz uses in-memory cache by default.
CONTENTBLITZ_CACHE_BACKEND=sqlite
CONTENTBLITZ_CACHE_TTL_SECONDS=1800
CONTENTBLITZ_CACHE_SQLITE_PATH=.tmp/contentblitz_cache.sqlite3
```

## Validation and Testing

Safe Phase 2 validation (non-live):

```bash
python scripts/validate_phase2.py
```

Unit and integration suite:

```bash
pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing
```

Live tests should skip when flags are off:

```bash
pytest tests/live -rs
```

Optional live smoke:

```bash
python scripts/dev/smoke_phase2_live.py --dry-run
CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_generate_text.py -s -rs
CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_search_web.py -s -rs
CONTENTBLITZ_RUN_LIVE_TESTS=1 CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=1 pytest tests/live/test_live_generate_image.py -s -rs
```

Note: live smoke tests are optional and environment/network dependent. This README does not claim successful live provider execution.

## Key Docs

- `docs/ContentBlitz_Execution_Spec.md`
- `docs/PHASE2_INTEGRATIONS.md`
- `docs/PROVIDER_CONTRACTS.md`
- `docs/CACHE_ARCHITECTURE.md`
- `docs/COST_CONTROLS.md`
- `docs/TESTING_STRATEGY.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/PHASE2_LIVE_SMOKE_TESTS.md`
