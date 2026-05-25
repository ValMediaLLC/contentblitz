# Performance Architecture

## Scope

This document describes how ContentBlitz reports and validates performance metadata in deterministic, mocked test flows.

Source of truth:

- `docs/ContentBlitz_Execution_Spec.md`
- `contentblitz/agents/research_agent.py`
- `contentblitz/workflow/graph.py`
- `frontend/components/result_view.py`

This document does not claim live provider speedups.

## Performance Model

ContentBlitz tracks performance at two levels:

- Node execution timing (`duration_ms`, node start/end timestamps)
- Provider timing metadata (when provider-backed calls occur)

The UI performance summary is a read-only aggregation of safe metadata emitted during workflow execution.

## Async Research Behavior

`research_agent_node` runs a deterministic async search fan-out:

1. Generate 3-5 queries (text model policy applied).
2. Execute provider calls concurrently with bounded semaphore concurrency.
3. Reconstruct results in original query order before downstream synthesis.
4. Optionally run fallback-provider queries when primary snippets are degraded/unusable.

Determinism guarantees:

- visible query order matches planned query order
- merge behavior is stable even when provider responses complete out of order
- fallback path preserves stable ordering for executed queries

## Deterministic Merge Ordering

Merge behavior remains deterministic across fan-out/fan-in by reducer design:

- `merge_source_entries`: stable order with dedupe identity rules
- `merge_progress_events`: stable append order with duplicate suppression
- `merge_cost_controls`: monotonic counters and strict-cap merges
- `merge_ui_node_statuses`: fixed status precedence resolution

Research fan-out also explicitly reorders async task results to the original query list before state write.

## Timing Metadata Fields

Research node metadata emitted in `research_data` includes:

- `provider_latency_total_ms`
- `provider_latency_wall_ms`
- `provider_latency_by_provider_ms`
- `provider_call_count`
- `provider_call_count_by_provider`
- `provider_timeout_count`
- `provider_timeout_count_by_provider`
- `search_provider_wall_timeout_ms`
- `search_provider_wall_timeout_triggered`
- `cache_hit`
- `fallback_used`

Workflow/UI safe timing metadata also includes per-node `duration_ms` and optional provider/model labels.

Notes:

- Aggregate provider time can exceed wall-clock duration in concurrent fan-out scenarios.
- Wall-clock totals should be interpreted separately from summed provider timings.

## Cost-Control Interaction

Cost controls are agent-owned:

- `search_queries_used_this_session` is incremented by agent logic, not provider payloads.
- `image_generations_used_this_session` is incremented by agent logic.
- `tokens_used_this_session` is updated from normalized usage metadata via agent-owned helpers.

When caps/budgets are reached, behavior degrades safely (partial success) without mutating orchestration architecture.

## Cache Interaction

Research cache is read before provider fan-out:

- Cache hit:
  - provider calls are skipped
  - `cache_hit=True`
  - search counter does not increment for that run
- Cache miss:
  - provider fan-out executes
  - non-degraded payloads are cache-eligible
  - degraded payloads are not cached

## Mocked Performance Regression Tests

Primary mocked regression coverage:

- `tests/integration/test_phase5_performance_contracts.py`
- `tests/integration/test_phase5_performance_baseline.py`

Run only performance contract tests:

```bash
pytest tests/integration/test_phase5_performance_contracts.py tests/integration/test_phase5_performance_baseline.py
```

Run full unit/integration suite with coverage:

```bash
pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing
```

## Optional Live Performance Smoke

Optional live checks are manual and explicit opt-in only.

Examples:

```bash
CONTENTBLITZ_ENABLE_LIVE_CALLS=1 python scripts/dev/smoke_query_handler.py
python scripts/dev/smoke_phase2_live.py --dry-run
CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_generate_text.py tests/live/test_live_search_web.py -rs
```

These commands validate operational behavior, not benchmark-grade throughput guarantees.

## Limitations of Local Mocked Timings

- Mocked timings reflect local process scheduling and synthetic waits.
- They do not represent provider-side queueing, regional latency, or real quota contention.
- Local machine load can shift absolute timing values.
- Contract tests validate shape, ordering, and safety semantics, not absolute performance targets.
