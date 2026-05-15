# Reducer Merge Stability

## Scope

This document describes reducer-based fan-out merge behavior in the LangGraph workflow state.

Primary file:

- `contentblitz/workflow/graph.py`

## Why Reducers Matter

Parallel branches (blog/linkedin/image/research/export paths) can emit updates in the same step.
Reducers provide deterministic reconciliation and prevent branch updates from wiping unrelated state.

## Reducer-Covered Fields

Current reducer-annotated fields in `WorkflowState` include:

- `content_drafts` via `merge_content_drafts`
- `draft_status` via `merge_draft_status`
- `sources` via `merge_source_entries`
- `image_prompts` via `merge_unique_text_list`
- `image_outputs` via `merge_image_outputs`
- `quality_scores` via `merge_nested_dict_skip_none`
- `retry_counts` via `merge_retry_counts`
- `errors` via `merge_error_entries`
- `export_metadata` via `merge_export_metadata`
- `cost_controls` via `merge_cost_controls`
- `status_messages` via `merge_unique_text_list`
- `warnings` via `merge_unique_text_list`
- `progress_events` via `merge_progress_events`
- `ui_node_statuses` via `merge_ui_node_statuses`

## Deterministic Merge Rules

Selected implemented rules:

- text-list reducers skip non-string, empty, `None`, and `null` placeholders
- source entries dedupe with stable ordering
- image outputs dedupe while stripping base64/data-image payloads
- nested dict merge skips `None` overwrites
- retry counts use per-key max semantics (not sum)
- progress events dedupe exact duplicates and preserve deterministic ordering
- UI node status precedence preserves terminal severity:
  - `failed > degraded > completed > skipped > running > pending`

## Retry and Cost Counter Semantics

- Retry counts are controlled by retry-router logic and merged deterministically.
- Cost control counters are merged to avoid invalid parallel overwrite conflicts.
- `budget_exceeded` uses sticky-true merge behavior.

## Fan-Out Safety Outcomes

Expected behavior under branch fan-out:

- text outputs are not blocked by recoverable image failures
- export metadata merges across formats without erasing prior successful paths
- status/warning streams remain deduplicated and deterministic
- merge behavior avoids `InvalidUpdateError` from conflicting parallel updates

## Verification Coverage

Reducer/fan-out behavior is covered by:

- `tests/unit/test_state_reducers.py`
- `tests/unit/test_workflow_routing.py`
- `tests/integration/test_reducer_merge_workflow.py`
