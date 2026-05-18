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
