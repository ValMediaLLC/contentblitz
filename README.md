# ContentBlitz

Deterministic multi-agent content generation system built with LangGraph.

ContentBlitz orchestrates specialized AI agents to generate blogs, LinkedIn posts, research reports, and image prompts through a state-safe, test-driven workflow architecture.

---

# Core Features

- Deterministic LangGraph orchestration
- Explicit global state model
- Modular multi-agent architecture
- Retry routing and quality validation
- Cost-control and token tracking
- Source-aware research pipeline
- Parallel content generation workflows
- Stateless tool execution
- Test-driven implementation
- Production-oriented architecture

---

# Architecture Overview

```text
START
  ↓
query_handler_node
  ↓
research_agent_node
  ↓
content_strategist_node
  ↓
parallel:
  ├── blog_writer_node
  ├── linkedin_writer_node
  └── image_agent_node
  ↓
quality_validator_node
  ↓
retry_router_node
  ↓
output_assembler_node
  ↓
export_node
  ↓
END
```

---

# Project Structure

```text
contentblitz/
├── contentblitz/
│   ├── agents/
│   ├── core/
│   ├── tools/
│   ├── workflow/
│   ├── config.py
│   └── state.py
│
├── docs/
│   └── ContentBlitz_Execution_Spec.md
│
├── scripts/
│   └── validate_phase1.py
│
├── tests/
│   ├── integration/
│   └── unit/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Development Environment

## Requirements

- Python 3.10+
- Git
- Virtual environment (.venv)

---

# Setup

## 1. Clone Repository

```bash
git clone https://github.com/ValMediaLLC/contentblitz.git
cd contentblitz
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=
SERP_API_KEY=
PERPLEXITY_API_KEY=
```

---

# Running Validation

## Structural Validation

```bash
python scripts/validate_phase1.py
```

---

## Unit + Integration Tests

```bash
pytest tests/unit tests/integration
```

---

## Coverage

```bash
pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing
```

---

# Current Status

## Completed

- Global state model
- LangGraph workflow graph
- Deterministic routing system
- Retry policy architecture
- Validation framework
- Unit test foundation
- Integration test foundation
- Repository + CI-ready structure

---

## In Progress

- Agent-by-agent implementation
- Tool integrations
- Export system
- UI layer

---

# Engineering Principles

ContentBlitz enforces:

- deterministic routing
- explicit state mutation
- stateless tools
- retry-safe workflows
- test-driven implementation
- no implicit cross-agent behavior

---

# Testing Philosophy

Every implementation step is:

1. implemented in isolation
2. validated with pytest
3. integration-tested
4. committed independently

---

# License

MIT License

---

# Author

ValMedia LLC