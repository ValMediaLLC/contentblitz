# ContentBlitz

ContentBlitz is a deterministic multi-agent content orchestration system built with LangGraph.

It orchestrates classification, research, strategy, writing, quality validation, retry routing, output assembly, and export through explicit state updates and deterministic routing rules.

## Current Project State (May 8, 2026)

- Phase 1 scope is implemented end-to-end for orchestration.
- 12 workflow nodes are implemented and wired in the graph.
- Test status: `233 passed` (`pytest -q`).
- Coverage status: `92% total` (`coverage report -m`).
- External providers are intentionally stubbed in this phase (no live network API calls by default).

## What Is Implemented

- Deterministic LangGraph workflow with conditional and fan-out routing.
- Explicit global state model with merge reducers for parallel writer updates.
- Query classification with deterministic fallback behavior.
- Research agent with cache-first behavior, source dedupe, and degraded fallback summaries.
- Blog and LinkedIn draft generation with retry feedback integration.
- Image path orchestration with prompt enhancement and recoverable failure handling.
- Quality validator scoring with best-draft tracking.
- Retry router with per-agent and session-level retry caps.
- Output assembler with source dedupe and partial-success handling.
- Export node with deterministic format handling (`markdown`, `html`, `pdf` fallback behavior).
- Extensive unit and integration test coverage for routing and prompt regression scenarios.

## What Is Intentionally Stubbed In Phase 1

- `contentblitz/tools/text.py` returns deterministic scaffold payloads (no live OpenAI calls).
- `contentblitz/tools/web_search.py` returns deterministic empty-result payloads (no live SERP/Perplexity calls).
- `contentblitz/tools/image.py` returns deterministic placeholder payloads (no live image generation calls).
- Cache and exports are state-driven/deterministic scaffolds, not production persistence/output I/O pipelines.

## Workflow Overview

```text
START
  -> query_handler_node
     -> clarification_node (if clarification_needed) -> END
     -> image_agent_node (image-only route)
     -> research_agent_node (if research_required)
     -> content_strategist_node
          -> blog_writer_node
          -> linkedin_writer_node
          -> image_agent_node (if requested)
  -> quality_validator_node
     -> retry_router_node (if retry_needed)
     -> output_assembler_node
  -> export_node (if export_requested)
  -> END
```

## Repository Layout

```text
contentblitz/
  contentblitz/
    agents/        # 12 workflow nodes + compatibility wrappers
    core/          # retry utility helpers
    tools/         # deterministic Phase 1 tool interfaces
    workflow/      # graph wiring + routing logic
    config.py
    state.py
  docs/
    ContentBlitz_Execution_Spec.md
    TESTING_STRATEGY.md
    KNOWN_LIMITATIONS.md
    PHASE1_REVIEW_SUMMARY.md
    RETRY_ROUTER_ARCHITECTURE.md
  scripts/
    validate_phase1.py
    dev/
      smoke_query_handler.py
      force_retry_scenarios.py
  tests/
    unit/
    integration/
  requirements.txt
  README.md
```

## Setup

### 1. Clone

```bash
git clone https://github.com/ValMediaLLC/contentblitz.git
cd contentblitz
```

### 2. Create and Activate Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=
SERP_API_KEY=
PERPLEXITY_API_KEY=
```

Note: these are part of the target architecture, but current Phase 1 tool modules are deterministic stubs.

## Run Validation and Tests

Structural validation:

```powershell
$env:PYTHONUTF8='1'; python scripts\validate_phase1.py
```

Unit + integration tests:

```bash
pytest tests/unit tests/integration
```

Quick full run:

```bash
pytest -q
```

Coverage:

```bash
pytest --cov=contentblitz --cov-report=term-missing
```

## Development Smoke Utilities

End-to-end workflow smoke run:

```powershell
python scripts/dev/smoke_query_handler.py
```

Forced retry scenarios:

```powershell
python scripts/dev/force_retry_scenarios.py
```

## Minimal Programmatic Usage

```python
from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph

graph = build_langgraph()
state = create_initial_state(
    user_query="Write a blog and LinkedIn post about AI workflow automation."
)
result = graph.invoke(state)

print(result["workflow_status"])
print(result["final_response"])
```

## Key Documentation

- `docs/ContentBlitz_Execution_Spec.md` - canonical architecture and constraints
- `docs/TESTING_STRATEGY.md` - testing layers and release-gate philosophy
- `docs/KNOWN_LIMITATIONS.md` - current limitations and planned improvements
- `docs/RETRY_ROUTER_ARCHITECTURE.md` - retry ownership and safety model
- `docs/PHASE1_REVIEW_SUMMARY.md` - phase review and readiness summary

## License

MIT License

## Author

ValMedia LLC
