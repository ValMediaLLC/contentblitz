# Observability Configuration

## Scope

ContentBlitz Phase 4 observability support is optional and environment-driven.

Tracing uses safe defaults:

- tracing is disabled by default
- LangSmith credentials are not required for normal startup
- LangSmith credentials are not required for unit/integration tests
- observability helpers never return or log API key values

## Environment Variables

ContentBlitz reads the following variables:

- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_PROJECT`
- `CONTENTBLITZ_TRACE_SAMPLE_RATE`
- `CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE`

## Defaults

If not provided:

- `LANGSMITH_TRACING` defaults to disabled
- `LANGSMITH_ENDPOINT` defaults to `https://api.smith.langchain.com`
- `LANGSMITH_PROJECT` defaults to `ContentBlitz`
- `CONTENTBLITZ_TRACE_SAMPLE_RATE` defaults to `1.0`
- `CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE` defaults to `1.0`

Sampling notes:

- values must be in `[0.0, 1.0]`
- invalid values fall back safely to defaults
- sampling decisions are deterministic by `session_id` when present
- sampling is observability-only and never affects workflow execution

If `LANGSMITH_TRACING=true` but `LANGSMITH_API_KEY` is missing:

- ContentBlitz degrades safely
- tracing remains disabled
- app startup and test execution continue normally

## Example Setup (Opt-In)

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<set-in-your-local-env-only>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=ContentBlitz
```

Do not commit API keys in `.env.example`, tests, fixtures, docs, logs, or state payloads.

## Programmatic Helpers

Use `contentblitz.core.observability`:

- `build_observability_config()` for secret-safe status
- `is_tracing_enabled()` for a simple boolean
- `observability_summary()` for a secret-safe dictionary payload

These helpers are read-only and do not mutate workflow state.

## UI Status And Diagnostics

The Streamlit frontend shows a UI-safe observability panel with:

- tracing status: `Enabled`, `Disabled`, or `Degraded`
- tracing enabled boolean (`true`/`false`)
- configured project name (safe label only)
- endpoint hostname only (no URL path/query)
- last trace attempt status (`Ready`, `Not requested`, or `Unavailable`)
- a safe note when tracing is unavailable
- an instruction to review the LangSmith dashboard manually

The UI diagnostics intentionally do **not** display:

- API key values
- full environment dumps
- raw LangSmith client internals
- raw provider payloads
- stack traces

Frontend diagnostics are read-only and do not mutate orchestration state. The UI
does not call providers or LangSmith directly.

## Execution Integration

LangGraph execution is traced via wrappers in `contentblitz.workflow.graph`:

- workflow-level spans around `invoke`, `stream`, `ainvoke`, and `astream`
- node-level spans remain canonical via LangGraph/LangSmith native graph tracing
- custom tool-level child spans are emitted from provider tools:
  - `generate_text`
  - `search_web`
  - `generate_image`
  - `cache_lookup`
  - `cache_write`

Tracing wraps execution only. It does not replace orchestration logic or routing.

### Duplicate Node Span Cleanup

When LangSmith tracing is enabled, custom node spans are intentionally disabled in
the LangSmith tracer adapter to avoid duplicate node spans with identical names
and near-identical durations. Tool spans remain enabled and attach beneath the
active workflow/node context.

## Safe Metadata Contract

Trace metadata is restricted to safe fields:

- `session_id` (if present)
- `workflow_status`
- `requested_outputs`
- `routing_decision`
- `node_name`
- `node_status`
- `tool_name`
- `provider`
- `model`
- `agent_key`
- `degraded`
- `fallback_used`
- `fallback_provider`
- `fallback_model`
- `fallback_reason`
- `input_token_count`
- `output_token_count`
- `total_token_count`
- `result_count`
- `citation_available_count`
- `image_url_present`
- `cache_hit`
- `cache_miss`
- `retry_attempt`
- `retry_exhausted`
- `budget_exceeded`
- `duration_ms`
- `observability_summary`:
  - `tracing_enabled`
  - `provider` (`langsmith`)
  - `project_name`
  - `endpoint_host`
- retry count summary
- cost counter summary
- export formats requested
- provider degraded (boolean)
- source count
- image output count
- boolean flags such as:
  - `research_required`
  - `clarification_needed`
  - `export_requested`
  - degraded/recoverable/export-failure status flags

The observability layer intentionally excludes:

- API keys and environment secrets
- raw environment-variable metadata keys such as:
  - `LANGSMITH_TRACING`
  - `LANGSMITH_ENDPOINT`
  - `LANGSMITH_PROJECT`
  - `LANGSMITH_API_KEY`
  - any `*_API_KEY`
- raw prompts and raw provider responses
- raw provider payload objects
- stack traces
- base64 image payloads
- full raw user queries
- full raw generated drafts/final responses

## Payload Reduction Strategy

Trace metadata intentionally summarizes large workflow payloads:

- `content_drafts` -> per-channel summaries (length, counts, version, preview, hash prefix)
- `final_response` -> summary (length, counts, section count, preview, hash prefix)
- `research_data` -> status/count summaries and short preview only
- `sources` -> counts and bounded domain summaries
- `image_outputs` -> counts, provider summary, and degraded counts
- `errors` -> bounded normalized safe error summaries

Full workflow state is not sent as trace metadata.

## Environment Metadata Safety

ContentBlitz never emits raw `.env` key/value metadata in spans.

Instead, spans include a safe observability summary:

```json
{
  "observability_summary": {
    "tracing_enabled": true,
    "provider": "langsmith",
    "project_name": "ContentBlitz",
    "endpoint_host": "api.smith.langchain.com"
  }
}
```

Notes:

- `endpoint_host` is hostname-only (no path/query).
- LangSmith/OpenAI/SERP/Perplexity API keys are never included.

## Redaction Layer

`contentblitz.core.redaction` provides a dedicated trace redaction pipeline:

- redact API-key-like tokens
- redact bearer tokens
- redact environment variable values
- redact stack-trace-like content
- redact base64 image payloads
- truncate oversized strings and collections
- normalize raw errors into safe summaries

Observability metadata is sanitized recursively through this layer before being
sent to tracing backends.
