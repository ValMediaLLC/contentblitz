# Known Limitations

## Scope

This document tracks known implementation and operational limits in the current Phase 2 codebase.

## Provider/Network Variability

Live provider behavior depends on external network and provider availability.

Implications:

- optional live tests may fail even when unit/integration tests are healthy
- degraded provider responses can appear during transient outages
- CI-safe coverage focuses on mocked deterministic tests

## Cache Backend Is Process-Local

Research caching currently uses an in-memory process store.

Implications:

- cache is not shared across process restarts
- no persistent datastore is used
- horizontal scaling cache coherence is not implemented

## Counter Merge Edge Case in Parallel Fan-Out

Cost control updates use dictionary merge reducers for parallel node writes.

Known edge case:

- in some multimodal fan-out flows, last-write behavior can undercount a subset of counters
- tests validate deterministic behavior, but this remains a known accounting limitation

## Query Classification Bias

Deterministic fallback classification remains conservative.

Known behavior:

- some prompts can over-select outputs (for example, certain LinkedIn requests may also select blog output)

## Research Quality in Degraded Paths

When providers return limited/unusable snippets:

- research agent returns structured degraded payloads
- summaries may fall back to deterministic synthesis language
- directional quality can be lower than fully cited SERP-backed paths

## No Persistent Storage Layer

Workflow state and cache are in-memory only.

Not implemented:

- database persistence
- durable artifact storage
- distributed cache invalidation

## Optional Live Tests Are Not CI Gate

`tests/live` are intentionally optional and skip-gated by flags.

Implications:

- live smoke checks are useful for manual operational validation
- release safety is enforced primarily through unit/integration contract suites

## LangGraph Warning

A LangGraph serializer deprecation warning related to `allowed_objects` may appear during test runs.

Current impact:

- does not affect test pass/fail logic
- should be resolved in a future dependency/serializer cleanup
