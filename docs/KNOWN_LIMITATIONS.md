# Known Limitations

## Scope

This document tracks known implementation and operational limits in the current codebase, including Phase 4 observability support.

## Provider/Network Variability

Live provider behavior depends on external network and provider availability.

Implications:

- optional live tests may fail even when unit/integration tests are healthy
- degraded provider responses can appear during transient outages
- CI-safe coverage focuses on mocked deterministic tests

## Lightweight Guardrails (Not Full Moderation)

Prompt-injection detection, output sanitization, citation validation, and export validation are deterministic and rule-based.

Implications:

- protection is intentionally lightweight and explainable
- advanced adversarial safety coverage is out of scope
- false positives/false negatives are still possible in edge cases

## Cache Backend Defaults and Operational Scope

Default cache is in-memory.

Implications:

- cache resets on process restart unless SQLite backend is enabled
- SQLite backend is local-only and not intended as distributed cache infrastructure
- no distributed cache invalidation is implemented

## Query Classification Bias

Deterministic classification/routing remains conservative.

Known behavior:

- some prompts can over-select outputs (for example, certain LinkedIn requests may also select blog output)

## Research Quality in Degraded Paths

When providers return limited/unusable snippets:

- research agent returns structured degraded payloads
- summaries may fall back to deterministic synthesis language
- directional quality can be lower than fully cited SERP-backed paths

## Local Persistence Is File-Based

Run persistence uses local JSON files (`.contentblitz_sessions` by default).

Implications:

- not a multi-user database-backed persistence layer
- intended for local/developer usage in current scope
- restore is read-only (no workflow rerun), but underlying exported files can be missing locally

## Optional Live Tests Are Not CI Gate

`tests/live` are intentionally optional and skip-gated by flags.

Implications:

- live smoke checks are useful for manual operational validation
- release safety is enforced primarily through unit/integration contract suites

## LangSmith Tracing Is Optional and Credential-Gated

Phase 4 observability support is additive and opt-in.

Current behavior:

- tracing is disabled unless `LANGSMITH_TRACING` is explicitly truthy
- if tracing is requested but `LANGSMITH_API_KEY` is missing, tracing degrades to disabled
- app startup, unit tests, and integration tests continue without LangSmith credentials
- tracing is best-effort; tracer setup/runtime failures degrade to no-op and never fail workflow execution

Sampling behavior:

- trace sampling is deterministic per session when `session_id` is available
- sampling is telemetry-only and never changes routing, retries, cost controls, or outputs
- when only failure sampling is enabled, child span coverage may be reduced for unsampled successful runs
- observability metadata intentionally strips raw env-var-style keys and exposes only safe summary labels (for example `endpoint_host`, not full endpoint URL)

Privacy/safety scope:

- observability metadata is redacted and summarized, but traces are still operational telemetry and not a compliance boundary
- no raw API keys, raw prompts, raw provider payloads, stack traces, or base64 image payloads should be present in traces
- if tracing setup fails, ContentBlitz intentionally degrades to no-op tracing without blocking workflow execution
- CI does not run live LangSmith validation; optional live smoke must be run manually when needed

## Export Validation Can Mark Individual Formats Failed

Export validation is format-specific and non-blocking per format.

Implications:

- one failed format does not necessarily block other requested formats
- workflows can still complete with partial export success
- failed export statuses are expected behavior in certain malformed/sanitized payload scenarios

## Synced-Folder File Operation Nuances

On some Windows/OneDrive setups, atomic file replace/delete behavior can be inconsistent for temporary files.

Implications:

- local persistence and cleanup operations may occasionally require retries
- validator logic includes safe fallback handling to avoid false negative readiness results

## LangGraph Warning

A LangGraph serializer deprecation warning related to `allowed_objects` may appear during test runs.

Current impact:

- does not affect test pass/fail logic
- should be resolved in a future dependency/serializer cleanup
