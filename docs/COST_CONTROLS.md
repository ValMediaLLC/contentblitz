# Cost Controls

## Scope

Cost controls are implemented with agent-owned counters, deterministic guard logic, and Phase 3 UI-safe visibility.

Core module:

- `contentblitz/core/cost_controls.py`

Primary agent integrations:

- `query_handler.py`
- `research_agent.py`
- `content_strategist.py`
- `blog_writer.py`
- `linkedin_writer.py`
- `image_agent.py`
- `retry_router.py`
- `output_assembler.py` (budget notice rendering)
- `contentblitz/ui/rendering.py` (safe usage summary derivation)
- `frontend/components/result_view.py` (usage display)

## Counters

`state["cost_controls"]` includes:

- `tokens_used_this_session`
- `search_queries_used_this_session`
- `image_generations_used_this_session`
- `total_retries_used_this_session`
- `budget_exceeded`

Session caps/thresholds:

- `token_budget_per_session` (default 10000)
- `search_query_cap_per_session` (default 5)
- `image_generation_cap_per_session` (default 3)
- `max_total_retries_per_session` (default 3)
- near-token threshold ratio: `0.90`

## Decision Rules

### Tokens

- Token usage is incremented from normalized text-tool usage (`total_tokens`)
- Near token budget uses `gpt-4o-mini` preference
- Exceeded budget sets `budget_exceeded=True`
- Query handler can route to safe error handling when budget is exceeded

### Search

- Research agent checks search cap before provider calls
- When cap reached:
  - search calls are skipped
  - degraded research metadata is returned
  - flow can continue as partial success

### Image

- Image agent checks image cap before generation
- When cap reached:
  - image generation is skipped
  - recoverable warning/error metadata is written

### Retries

- Retry router enforces per-agent retry policy and total session retry cap
- Retry fan-out is reduced or stopped when caps are reached

## Phase 3 UI Visibility

UI usage summary derives safe counters from `cost_controls` and `usage_metrics` when available.

Budget visibility states:

- `normal`
- `degraded`
- `limited`
- `budget_exceeded`

Displayed values remain aggregate-only and do not expose provider billing payloads.

## Ownership Rules

- Tools are stateless and do not update counters.
- Agents own counter updates and budget decisions.
- Output assembler surfaces budget warnings in final response.

## Known Limitation

Parallel fan-out merge semantics use deterministic reconciliation, but additive interpretation across concurrent branch updates can still differ from serialized single-branch accounting in edge cases.

## Security Rules

- `.env` is never committed.
- API keys are read only from environment variables.
- Tools are stateless.
- State never stores secrets.
- Provider errors are normalized.
- Base64 image data is never stored in state.
