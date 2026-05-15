# Session Persistence

## Scope

This document describes local session persistence, serialization safety, and restore behavior in Phase 3.

Primary files:

- `contentblitz/persistence/session_store.py`
- `contentblitz/persistence/serialization.py`
- `contentblitz/persistence/models.py`
- `frontend/session.py`
- `frontend/pages/history.py`

## Storage Model

- Backend: local JSON files.
- Default directory: `.contentblitz_sessions/`.
- One persisted workflow run per file (`<run_id>.json`).
- Directory can be overridden with `CONTENTBLITZ_SESSION_DIR`.

## Serialization Behavior

Persisted records include sanitized workflow data such as:

- run/session IDs and timestamps
- prompt/query and requested outputs
- workflow status (`workflow_status`, `ui_workflow_status`)
- content drafts and partial outputs
- image prompts and sanitized image outputs
- sources (deduplicated/sanitized)
- quality/citation summary metadata
- cost/usage aggregate counters
- export metadata
- progress events
- node status map
- prompt-injection signal metadata

## Serialization Safety Rules

Persistence excludes or normalizes:

- API keys/env secrets
- raw provider payloads
- stack traces/internal exception dumps
- base64/data-image content
- unsafe prompt/system/developer leakage
- misleading progress metadata fields that conflict with final status

## Restore Behavior

Restore behavior in `History` page:

- loads persisted run safely
- hydrates UI with restored result payload
- does not rerun workflow
- does not rerun providers
- does not regenerate exports

If saved export file paths are missing locally, restore keeps run data and adds safe warning text.

## UI Session State Isolation

Frontend session keys are namespaced with `cbx_ui_*` and managed in `frontend/session.py`.

This keeps UI-local state separate from orchestration-owned workflow state.

## Timeline and History

- Browser-session timeline is tracked in frontend session state.
- Persisted run history is tracked in local session store.
- Restoring a run updates current UI view without mutating persisted record content.
