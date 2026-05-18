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

## Defaults

If not provided:

- `LANGSMITH_TRACING` defaults to disabled
- `LANGSMITH_ENDPOINT` defaults to `https://api.smith.langchain.com`
- `LANGSMITH_PROJECT` defaults to `ContentBlitz`

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

## Execution Integration

LangGraph execution is traced via wrappers in `contentblitz.workflow.graph`:

- workflow-level spans around `invoke`, `stream`, `ainvoke`, and `astream`
- node-level spans around each authoritative node function

Tracing wraps execution only. It does not replace orchestration logic or routing.

## Safe Metadata Contract

Trace metadata is restricted to safe fields:

- `session_id` (if present)
- `workflow_status`
- `requested_outputs`
- `routing_decision`
- `node_name`
- `node_status`
- retry count summary
- cost counter summary
- export formats requested
- boolean flags such as:
  - `research_required`
  - `clarification_needed`
  - `export_requested`
  - degraded/recoverable/export-failure status flags

The observability layer intentionally excludes:

- API keys and environment secrets
- raw prompts and raw provider responses
- raw provider payload objects
- stack traces
- base64 image payloads
