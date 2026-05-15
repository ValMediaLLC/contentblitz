# Phase 3 UI Architecture

## Scope

This document describes the current Streamlit UI shell and how it integrates with orchestration, rendering, and persistence layers.

## UI Entry and Routing

Primary files:

- `frontend/app.py`
- `frontend/router.py`
- `frontend/config.py`
- `frontend/theme.py`

Behavior:

- `streamlit run frontend/app.py` is the supported launch path.
- App startup initializes frontend session keys only.
- No provider calls are made at UI startup.
- Routing uses segmented navigation across:
  - Run Workflow
  - History
  - About

## Run Workflow Page

Primary file:

- `frontend/pages/run_workflow.py`

Behavior:

- Collects prompt + output controls (`blog`, `linkedin`, `research`, `image`).
- Export formats are selectable only when `Enable Export` is checked.
- Submission validation blocks empty prompt, empty output set, and export-without-format.
- UI options are passed to orchestration service layer only.
- Orchestrator remains authoritative for final routing/classification.

## History and Restore

Primary files:

- `frontend/pages/history.py`
- `frontend/session.py`
- `contentblitz/persistence/serialization.py`
- `contentblitz/persistence/session_store.py`

Behavior:

- Lists saved runs from local persistence.
- Restores sanitized persisted outputs/status into UI state.
- Restore is read-only:
  - does not rerun workflow
  - does not rerun providers
  - does not regenerate exports
- Missing local export files are surfaced as safe warnings.

## Rendering Pipeline

Primary files:

- `contentblitz/ui/rendering.py`
- `contentblitz/ui/status.py`
- `contentblitz/ui/progress.py`
- `contentblitz/ui/error_display.py`
- `frontend/components/result_view.py`

Behavior:

- Rendering helpers are read-only over workflow state.
- UI status/progress is normalized into safe display payloads.
- Partial outputs render only after appropriate node completion/degraded states.
- Image output display strips base64/data-image payloads.
- Sources are deduplicated and sanitized for display.
- Errors/warnings are normalized before rendering.

## Usage and Cost Visibility

Usage is displayed from safe aggregate counters, not billing APIs.

Displayed metrics include:

- estimated tokens in/out
- text generation calls
- search queries
- sources returned
- image requests and image failures
- retry attempts
- degraded operations
- export generation count
- budget state (`normal`, `degraded`, `limited`, `budget_exceeded`)
- estimated workflow cost level (`low`, `medium`, `high`)

## State Ownership Rules

- UI must not mutate orchestration internals.
- Tools remain stateless.
- Agents/nodes own workflow state mutation.
- Frontend session state is isolated with `cbx_ui_*` keys.

## Authoritative Node Names

UI status/progress references the same 12 authoritative workflow nodes:

- `query_handler_node`
- `clarification_node`
- `research_agent_node`
- `content_strategist_node`
- `blog_writer_node`
- `linkedin_writer_node`
- `image_agent_node`
- `quality_validator_node`
- `retry_router_node`
- `output_assembler_node`
- `export_node`
- `error_handler_node`
