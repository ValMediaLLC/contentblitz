# ContentBlitz Observability (Phase 4)

## Purpose

Phase 4 adds optional LangSmith tracing to improve debugging and execution visibility for:

- workflow-level execution (`contentblitz_workflow`)
- node progression across the authoritative 12-node LangGraph
- provider/tool timing and fallback metadata

Tracing is additive and must never change orchestration behavior.

## Security and Safety Guarantees

- `.env` is never committed.
- `LANGSMITH_API_KEY` is read only from environment variables.
- tracing must not mutate workflow state.
- tracing must not alter routing.
- tracing must not alter retry counts.
- tracing must not alter cost counters.
- raw user input is not included in trace metadata.
- raw provider payloads are not included in trace metadata.
- base64 image data is never traced.
- secrets are redacted before metadata is sent.

## Environment Variables

Required for optional tracing:

- `LANGSMITH_TRACING`
- `LANGSMITH_API_KEY`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_PROJECT`

Optional Phase 4 controls:

- `CONTENTBLITZ_TRACE_SAMPLE_RATE`
- `CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE`
- `CONTENTBLITZ_RUN_LANGSMITH_SMOKE`

Defaults:

- tracing is disabled unless explicitly enabled
- `LANGSMITH_ENDPOINT` defaults to `https://api.smith.langchain.com`
- `LANGSMITH_PROJECT` defaults to `ContentBlitz`
- sample rates default to `1.0`

If `LANGSMITH_TRACING=true` but `LANGSMITH_API_KEY` is missing, tracing degrades safely to disabled without failing app startup, tests, or workflow runs.

## Local Setup (Optional)

Example local `.env` values:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<set-locally>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=ContentBlitz
CONTENTBLITZ_TRACE_SAMPLE_RATE=1.0
CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE=1.0
CONTENTBLITZ_RUN_LANGSMITH_SMOKE=0
```

Do not commit keys or secret values to git, tests, fixtures, or exported artifacts.

## Disabled-by-Default Behavior

- normal ContentBlitz startup does not require LangSmith credentials
- unit and integration tests do not require LangSmith credentials
- tracing setup/runtime failures degrade to safe no-op tracing
- provider execution remains independent from tracing availability

## What Is Traced

Safe, compact metadata is included for:

- workflow start/end status
- routing decision
- requested outputs
- root-trace `intent` derived only from resolved backend state:
  - `requested_outputs`
  - `export_formats_requested`
- node name and node status
- retry count and retry exhaustion summaries
- cost counter summaries
- export request summaries
- degraded and fallback flags
- source/image count summaries
- token count summaries from tools when available
- duration metadata for workflow and tool spans
- observability summary:
  - `tracing_enabled`
  - `provider` (`langsmith`)
  - `project_name`
  - `endpoint_host` (hostname only)

LangSmith root trace `Input` intentionally contains only safe resolved intent labels:

```json
{"intent": ["blog", "linkedin", "image", "pdf"]}
```

Intent labels are derived only from:

- `requested_outputs`
- `export_formats_requested` (or normalized export format metadata)

## What Is Never Traced

- API key values (`LANGSMITH_API_KEY`, `OPENAI_API_KEY`, `SERP_API_KEY`, `PERPLEXITY_API_KEY`)
- raw `.env` values or environment dumps
- raw prompts and raw provider payloads
- full raw generated drafts/final responses
- raw stack traces
- base64 image data
- raw user query text
- query previews (`query_preview`, `sanitized_user_query`, `user_query`)

## Redaction and Safe Metadata Policy

The redaction layer (`contentblitz.core.redaction`) sanitizes metadata before trace emission:

- key/token/bearer redaction
- stack trace detection and normalization
- base64 payload redaction
- raw payload field blocking
- bounded string/list/dict truncation
- JSON-serializable normalization

Large state fields are summarized (length/count/hash-prefix/short preview) instead of sending full payloads.

## Interaction With LangGraph

- tracing wraps graph execution via safe workflow-span wrappers in `contentblitz.workflow.graph`
- wrapper coverage includes `invoke`, `stream`, `ainvoke`, and `astream`
- custom node spans are no-op in the LangSmith adapter to avoid duplicate spans when native LangGraph node spans already exist
- graph architecture and authoritative node set are unchanged

## Interaction With Streamlit UI

The Workflow section renders a safe observability panel using `contentblitz.ui.observability`:

- status (`Enabled`, `Disabled`, `Degraded`)
- tracing enabled boolean
- safe project label
- endpoint hostname only
- last trace attempt label

The UI does not call LangSmith directly, does not call providers, and does not mutate orchestration state.

## Interaction With Provider Tools

Provider tools emit child spans when tracing is enabled:

- `generate_text`
- `search_web` (with provider fallback spans such as `serp` / `perplexity_fallback` when used)
- `generate_image` (with model fallback metadata when used)
- cache spans (`cache_lookup`, `cache_write`)

Tool metadata is sanitized through safe metadata builders and redaction before emission.

## Export Metadata Ownership Boundary

- `query_handler_node` may initialize/normalize export intent metadata:
  - `export_requested`
  - `export_metadata.formats_requested`
- `export_node` is the only node that writes execution-result export fields:
  - `export_metadata.export_paths`
  - `export_metadata.exported_at`
  - `export_metadata.error_log` entries from export execution

## Validation and Testing

Normal suite (no LangSmith credentials required):

```bash
pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing
```

Phase 4 validation runner:

```bash
python scripts/validate_phase4.py
```

Dry-run smoke (no LangSmith calls):

```bash
python scripts/dev/smoke_langsmith.py --dry-run
```

Optional live smoke (explicit opt-in only):

```bash
CONTENTBLITZ_RUN_LANGSMITH_SMOKE=1 python scripts/dev/smoke_langsmith.py
```

## Troubleshooting: No Traces Appearing

1. Confirm `LANGSMITH_TRACING=true`.
2. Confirm `LANGSMITH_API_KEY` is set in your environment.
3. Confirm `LANGSMITH_PROJECT`/`LANGSMITH_ENDPOINT` values are valid.
4. Confirm sample rates are not `0.0` for the path you expect.
5. Run `python scripts/dev/smoke_langsmith.py --dry-run` to validate safe config detection.
6. If tracing is degraded/disabled in UI, workflow execution is still expected to succeed without traces.

## Privacy and Security Limitations

- tracing is best-effort and telemetry-only; it is not a security boundary
- summarized previews may still contain normal business text, but secret patterns are redacted
- observability does not replace provider-side audit controls or external SIEM requirements
- optional live smoke is manual and not part of default CI validation
