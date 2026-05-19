# ContentBlitz Technical Debt Register

## Scope

This document tracks intentional non-blocking debt and future enhancements that
were identified during TODO cleanup and Phase 4 observability hardening.

Guidelines:

- keep architecture guarantees intact
- do not use this backlog to bypass test quality
- keep items actionable and testable
- prefer explicit categories over vague inline TODOs

## Architecture

### Deterministic Keyword Extraction Quality

- Source: `contentblitz/agents/research_agent.py`
- Status: partially resolved
- Resolved in current baseline:
  - bounded deterministic stopword filtering for extracted keywords
  - deterministic duplicate removal with stable ordering
  - regression tests for stopword removal and meaningful-term preservation
- Remaining scope:
  - optional bounded singular/plural normalization is deferred until additional
    deterministic regression coverage is added.
- Risk level: low (quality-only; no routing/state safety impact)

## Frontend UX

### Result View Module Decomposition

- Source: removed inline TODO from `frontend/components/result_view.py`
- Current behavior: module is large and handles multiple rendering concerns.
- Risk level: low to medium (maintainability risk, not runtime safety)
- Next action: split into focused modules for execution indicators, content
  panels, and export rendering without changing rendering payload/state rules.

## Observability Enhancements

### Optional Live Smoke Evidence Capture

- Source: Phase 4 release process
- Current behavior: dry-run smoke is validated by default; live smoke remains
  manual/opt-in.
- Risk level: low
- Next action: add an optional release checklist step to capture one successful
  live-smoke run artifact per release candidate.

### LangGraph Serializer Warning Follow-up

- Source: repeated warning in validation runs (`allowed_objects` deprecation)
- Current behavior: warning is non-blocking.
- Risk level: low (noise/maintenance)
- Next action: evaluate dependency update or explicit serializer configuration
  to remove warning while preserving behavior.

## Provider Optimization

### LinkedIn-Only Routing Precision

- Source: `contentblitz/agents/query_handler.py`
- Status: resolved
- Resolved in current baseline:
  - explicit LinkedIn-only phrasing no longer defaults to blog in fallback
    classification
  - `LinkedIn article` behavior is deterministic and LinkedIn-scoped unless blog
    is explicitly requested
  - `linked in` (space-separated) platform phrasing is treated as LinkedIn
  - targeted regression tests cover LinkedIn-only, blog-only, and multi-output
    prompt variants
- Risk level: low to medium (routing precision)
- Follow-up: continue monitoring prompt-regression suites for edge phrasing.

### Deterministic Image Fallback Asset Strategy

- Source: `contentblitz/agents/image_agent.py` (`TODO(provider)`)
- Current behavior: image failures are recoverable and surfaced safely, but no
  deterministic fallback image asset is produced yet.
- Risk level: low to medium (UX quality, not workflow safety)
- Next action: define safe fallback asset contract and add tests ensuring no
  base64 payloads or unsafe paths are introduced.

## Export Enhancements

### Partial Export Failure UX Clarity

- Source: known limitation in export validation behavior
- Current behavior: format-specific failures are recoverable and non-blocking.
- Risk level: low
- Next action: improve user-facing diagnostics for per-format failures while
  preserving recoverable export semantics.

## Caching Improvements

### Distributed/Persistent Cache Strategy

- Source: known limitation (in-memory/SQLite local scope)
- Current behavior: cache is process-local by default.
- Risk level: medium for scale scenarios
- Next action: evaluate Redis/managed cache option with identical cache key and
  safety semantics.

## Performance

### UI Rendering Hotspot Baseline

- Source: large Phase 3 result rendering paths
- Current behavior: functional but some paths render many sections/components.
- Risk level: low
- Next action: profile render time for large sessions and tune layout/helpers
  without changing workflow state ownership.

## Testing Gaps

### TODO-Driven Coverage Guardrails

- Source: TODO cleanup audit
- Current behavior: high-value TODOs are documented; targeted tests exist for
  observability, routing contracts, and state non-mutation.
- Risk level: low
- Next action: add focused regression coverage when each debt item is resolved;
  do not defer tests for safety-critical changes.

## Future Roadmap

### Post-Stabilization Enhancements

- advanced observability analytics and dashboards
- deeper deterministic routing heuristics for output-intent precision
- expanded export UX polish and error explainability
- scaled cache backend options for multi-process deployments

These roadmap items are intentionally deferred and are not blockers for current
release readiness.
