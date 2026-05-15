# ContentBlitz

ContentBlitz is a LangGraph-based multi-agent content orchestration system with a Phase 3 Streamlit UI, export pipeline, persistence/restore flow, and deterministic non-live validation.

## Current Implementation

- 12-node authoritative workflow graph is active.
- Provider-backed tools are integrated for:
  - OpenAI text generation (`gpt-4o` with `gpt-4o-mini` fallback)
  - SERP search with Perplexity fallback
  - OpenAI image generation (`dall-e-3` with `dall-e-2` fallback)
- Phase 3 UI shell is implemented:
  - Run Workflow page
  - History/restore page
  - About page
- Export formats implemented:
  - Markdown, HTML, PDF, DOCX
- Guardrails implemented:
  - prompt-injection detection/sanitization
  - output sanitization
  - citation/source validation
  - export payload validation
- Session persistence and restore are implemented with safe serialization.
- Unit/integration tests are deterministic and non-live by default.

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

Core provider and optional live-test flags:

```env
OPENAI_API_KEY=
SERP_API_KEY=
PERPLEXITY_API_KEY=
CONTENTBLITZ_RUN_LIVE_TESTS=0
CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=0
```

Cache configuration (optional; default backend remains in-memory if unset):

```env
CONTENTBLITZ_CACHE_BACKEND=sqlite
CONTENTBLITZ_CACHE_TTL_SECONDS=1800
CONTENTBLITZ_CACHE_SQLITE_PATH=.tmp/contentblitz_cache.sqlite3
```

UI/export/persistence directory overrides (optional):

```env
CONTENTBLITZ_EXPORT_DIR=exports
CONTENTBLITZ_SESSION_DIR=.contentblitz_sessions
```

## Security Baseline

- `.env` is never committed.
- API keys are read only from environment variables.
- Tools are stateless.
- State never stores secrets.
- Provider errors are normalized.
- Base64 image data is never stored in workflow state or persisted runs.

## Validation and Testing

Phase 3 validation (non-live, deterministic):

```bash
python scripts/validate_phase3.py --dry-run
```

Phase 2 validation (non-live, includes live-test skip gating checks):

```bash
python scripts/validate_phase2.py
```

Unit and integration suite:

```bash
pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing
```

Live tests are optional and skip by default:

```bash
pytest tests/live -rs
```

Optional live smoke:

```bash
python scripts/dev/smoke_phase2_live.py --dry-run
```

## Frontend Run Command

```bash
streamlit run frontend/app.py
```

UI startup does not require API keys and does not execute provider calls automatically.

## Key Docs

- `docs/ContentBlitz_Execution_Spec.md`
- `docs/PHASE3_UI_ARCHITECTURE.md`
- `docs/EXPORT_SYSTEM.md`
- `docs/VALIDATION_FRAMEWORK.md`
- `docs/GUARDRAILS_AND_SANITIZATION.md`
- `docs/SESSION_PERSISTENCE.md`
- `docs/REDUCER_MERGE_STABILITY.md`
- `docs/PHASE2_INTEGRATIONS.md`
- `docs/PROVIDER_CONTRACTS.md`
- `docs/CACHE_ARCHITECTURE.md`
- `docs/COST_CONTROLS.md`
- `docs/TESTING_STRATEGY.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/PHASE2_LIVE_SMOKE_TESTS.md`
