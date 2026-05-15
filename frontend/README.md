# ContentBlitz Frontend

This directory contains the Streamlit UI shell for ContentBlitz Phase 3.
The frontend is part of the main ContentBlitz project and uses the root Python
environment.

## Install

```bash
python -m pip install -r requirements.txt
```

No separate frontend dependency install is required.

## Run

```bash
streamlit run frontend/app.py
```

## Pages

- `Run Workflow`
  - prompt input
  - output controls (`blog`, `linkedin`, `research`, `image`)
  - export controls (`markdown`, `html`, `pdf`, `docx`)
  - progress + node status + result rendering
- `History`
  - persisted run list
  - restore into UI state
  - restore is read-only and does not rerun providers
- `About`
  - architecture boundaries and UI ownership notes

## Architecture Rules

- UI components call orchestration via `frontend/services/orchestrator_client.py`.
- No provider calls are made directly from frontend modules.
- No orchestration business logic is duplicated in pages/components.
- Session keys are frontend-prefixed and isolated in `frontend/session.py`.
- App startup must not require API keys or execute provider calls automatically.
- UI rendering is read-only over orchestration output.
- Cost/usage visibility uses safe aggregate counters only (no provider billing APIs).
