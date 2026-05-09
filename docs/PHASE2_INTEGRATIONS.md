# Phase 2 Integrations

## Scope

Phase 2 wires real provider-backed tools into the existing graph while preserving deterministic unit/integration testing.

Integrated tools:

- `contentblitz/tools/generate_text.py`
- `contentblitz/tools/search_web.py`
- `contentblitz/tools/perplexity.py`
- `contentblitz/tools/generate_image.py`

Compatibility adapters used by agents:

- `contentblitz/tools/text.py`
- `contentblitz/tools/web_search.py`
- `contentblitz/tools/image.py`

## Environment Variables

Provider tools read credentials from environment variables at invocation time:

- `OPENAI_API_KEY`
- `SERP_API_KEY`
- `PERPLEXITY_API_KEY`

Optional live test flags:

- `CONTENTBLITZ_RUN_LIVE_TESTS`
- `CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS`

## Provider Setup

### OpenAI Text (`generate_text`)

- Primary model: `gpt-4o`
- Fallback model: `gpt-4o-mini`
- Retries: `RETRY_POLICY[agent_key] + 1` per model
- Prompt safety guard is applied before provider call
- On full failure, returns degraded structured result

### SERP Search (`search_web`, provider=`serp`)

- Uses `SERP_API_KEY`
- Normalizes provider payload to `SearchWebResult`
- Deduplicates exact duplicate URLs
- Returns degraded structured result on failure/unusable payloads

### Perplexity Fallback (`search_web`, provider=`auto`)

- `auto` strategy:
  1. Run SERP first
  2. If SERP degraded/unusable, run Perplexity
- Perplexity entries can have `url=None`
- Citations are not invented when URLs are missing

### DALL-E Image (`generate_image`)

- Primary model: `dall-e-3`
- Fallback model: `dall-e-2`
- Prompt safety guard is applied before provider call
- Returns URL or provider file reference only
- Never returns base64

## Agent vs Tool Ownership

Tools:

- are stateless
- do not mutate workflow state
- do not increment cost counters
- return normalized result objects

Agents:

- own state mutation
- own cache read/write decisions
- own counter updates in `cost_controls`
- own degraded/partial-success routing behavior

## Failure Behavior

- Missing provider key: safe degraded result at invocation time
- Provider/network errors: normalized error payloads
- No raw stack traces in normalized tool errors
- Degraded results remain structured and recoverable

## Security Rules

- `.env` is never committed.
- API keys are read only from environment variables.
- Tools are stateless.
- State never stores secrets.
- Provider errors are normalized.
- Base64 image data is never stored in state.

## Live Validation

Optional live smoke tests exist under `tests/live` and are skip-gated by default.

This document does not claim successful live provider execution; live outcomes depend on credentials and network/provider availability.
