# CONTENTBLITZ EXECUTION SPEC (FINAL)

---
## 1. GLOBAL STATE MODEL

{
  "session_id": "string",
  "user_id": "string",
  "user_query": "string",

  "intent": "string",
  "routing_decision": "string",
  "requested_outputs": ["blog", "linkedin", "image", "research"],

  "conversation_history": [],

  "research_required": false,
  "clarification_needed": false,
  "clarification_message": null,

  "research_data": {},
  "sources": [],

  "content_brief": {
    "blog": {},
    "linkedin": {},
    "image": {}
  },

  "content_drafts": {
    "blog": {
      "body": "",
      "version": 0
    },
    "linkedin": {
      "body": "",
      "version": 0
    },
    "research_report": {
      "body": ""
    }
  },

  "best_drafts": {
    "blog": null,
    "linkedin": null
  },

  "attempt_history": {
    "blog": [],
    "linkedin": [],
    "image": []
  },

  "retry_feedback": {
    "blog": [],
    "linkedin": []
  },

  "retry_counts": {
    "query_handler": 0,
    "research_agent": 0,
    "content_strategist": 0,
    "blog_writer": 0,
    "linkedin_writer": 0,
    "image_agent": 0,
    "quality_validator": 0,
    "output_assembler": 0,
    "export": 0
  },

  "quality_scores": {},

  "image_prompts": [],
  "image_outputs": [],

  "tool_outputs": {},
  "errors": [],

  "final_response": "",
  "workflow_status": "",

  "export_requested": false,

  "export_metadata": {
    "formats_requested": [],
    "export_paths": {},
    "exported_at": null,
    "error_log": []
  },

  "cache_metadata": {
    "enabled": true,
    "ttl_seconds": 1800,
    "backend": "in_memory",
    "keys": []
  },

  "cost_controls": {
    "tokens_used_this_session": 0,
    "search_queries_used_this_session": 0,
    "image_generations_used_this_session": 0,
    "total_retries_used_this_session": 0,
    "budget_exceeded": false
  }
}

---

## 2. SYSTEM NODES

Total nodes: 12

- query_handler_node
- clarification_node
- research_agent_node
- content_strategist_node
- blog_writer_node
- linkedin_writer_node
- image_agent_node
- quality_validator_node
- retry_router_node
- output_assembler_node
- export_node
- error_handler_node

---

## 3. WORKFLOW GRAPH

START  
→ query_handler_node  

→ (clarification_needed → clarification_node → END)  

→ (research_required → research_agent_node)  
→ content_strategist_node  

→ parallel:  
  → blog_writer_node  
  → linkedin_writer_node  
  → image_agent_node (if requested)  

→ quality_validator_node  

→ retry_router_node (if needed)  

→ output_assembler_node  

→ (export_requested → export_node)  

→ END  

---

## 4. ROUTING RULES

- Deterministic routing first
- LLM only used for classification
- retry_counts increment BEFORE routing

---

## 5. MEMORY MODEL

- Short-term: full state object
- Conversation: conversation_history
- Optional long-term: brand + style

---

## 6. TOOL INTERFACES

generate_text(prompt, agent_key, model="gpt-4o")

Rules:
- retries = RETRY_POLICY[agent_key] + 1
- fallback = gpt-4o → gpt-4o-mini
- tool is stateless

search_web:
- primary: SERP API
- fallback: Perplexity

generate_image:
- primary: DALL-E 3
- fallback: DALL-E 2

---

## 7. CACHE SYSTEM

cache_key = "research:{hash}:{depth}"

Rules:
- SHA256 normalized query
- no raw user input
- exact match only

---

## 8. AGENTS (SUMMARY)

Query Handler:
- classify intent
- set routing

Research Agent:
- populate research_data

Content Strategist:
- create content_brief

Writers:
- generate drafts
- increment version before writing

Quality Validator:
- score outputs
- update best_drafts

Retry Router:
- increment retry_counts before routing

Output Assembler:
- deterministic assembly
- no LLM

Export Node:
- export final_response

---

## 9. OUTPUT ASSEMBLER

- builds final_response
- generates research report if needed
- deduplicates sources
- sets workflow_status

---

## 10. SOURCE RULES

- dedupe by URL
- fallback to title
- highest credibility wins

---

## 11. COST CONTROL

- agents update counters
- tools are stateless

---

## 12. RETRY POLICY

Total executions = 1 + RETRY_POLICY[key]

---

## 13. SECURITY

- no user input in system prompts
- injection guard enforced

---

## 14. CONSISTENCY RULES

- no implicit behavior
- no undeclared state access
- deterministic routing

---

## 15. ACCEPTANCE CHECKLIST

- all state fields defined
- all nodes implemented
- routing deterministic
- retry consistent

## 16. DEVELOPMENT ENVIRONMENT & PROJECT SETUP

This section defines the required development environment, dependency management, testing framework, and version control rules. These are mandatory and must be followed exactly.

---

### 16.1 Python Environment

- Python version: **3.10 or higher**
- A virtual environment MUST be used.

#### Setup:

    python -m venv .venv

Activate:

    # macOS / Linux
    source .venv/bin/activate

    # Windows
    .venv\Scripts\activate

---

### 16.2 Dependency Management

Dependencies MUST be managed via `requirements.txt`.

#### Required file:

    requirements.txt

#### Minimum dependencies:

    langgraph
    openai
    pytest
    pytest-cov
    python-dotenv

#### Installation:

    pip install -r requirements.txt

---

### 16.3 Testing Framework

- The system MUST use **pytest**
- The system MUST NOT use `unittest`

#### Rules:

- All tests must be written using pytest
- Test files must follow naming convention:

    test_*.py

- Tests must reside in:

    tests/unit/
    tests/integration/

#### Commands:

    pytest

With coverage:

    pytest --cov=contentblitz --cov-report=term-missing

---

### 16.4 Version Control (Git)

Git MUST be initialized before implementation begins.

#### Initialize:

    git init

---

### 16.5 .gitignore

The following file MUST be created:

    .gitignore

#### Contents:

    .venv/
    __pycache__/
    *.pyc
    .env
    .pytest_cache/
    coverage/
    docs/

---

### 16.6 Repository Structure (Initial)

The following structure MUST exist before agent implementation:

    contentblitz/
    ├── contentblitz/
    │   ├── agents/
    │   ├── tools/
    │   ├── workflow/
    │   ├── core/
    │   ├── config.py
    │   ├── state.py
    │
    ├── tests/
    │   ├── unit/
    │   ├── integration/
    │
    ├── docs/
    │   └── ContentBlitz_Execution_Spec.md
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

---

### 16.7 Environment Variables

A `.env` file MUST be used for secrets.

#### Required variables:

    OPENAI_API_KEY=
    SERP_API_KEY=
    PERPLEXITY_API_KEY=

#### Rules:

- `.env` MUST NOT be committed to Git
- All tools MUST read from environment variables only
- No secrets stored in code or state

---

### 16.8 Codex Execution Rules

All Codex prompts MUST assume:

- `.venv` exists and is active
- `requirements.txt` is the source of dependencies
- pytest is used for all testing
- the spec file exists at:

    docs/ContentBlitz_Execution_Spec.md

#### Hard constraints:

- Do NOT use `unittest`
- Do NOT install dependencies outside `requirements.txt`
- Do NOT modify architecture defined in spec
- Do NOT introduce new frameworks

---

### 16.9 Acceptance Criteria

Before proceeding to Phase 1:

- Virtual environment is created
- Dependencies install successfully
- Git repository initialized
- `.gitignore` is applied
- Project structure exists
- pytest runs successfully (even with empty tests)

---

# 🚀 IMPLEMENTATION PHASES

## PHASE 1: CORE ENGINE

Goal: Build working LangGraph system without integrations

Implement:

- state model
- workflow graph
- routing functions
- retry system
- agent skeletons (no real APIs)
- output assembler (deterministic)

Deliverable:
- system runs end-to-end with mock data

---

## PHASE 2: INTEGRATIONS

Goal: Make system production-capable

Implement:

- OpenAI (generate_text)
- SERP API (search_web)
- DALL-E (generate_image)
- Perplexity fallback
- caching layer
- cost controls

Deliverable:
- full real system with external APIs

---

## PHASE 3: UI + EXPORT

Goal: Make system user-facing and export-capable

Implement:

- Streamlit UI
- export system (PDF, HTML, Markdown, DOCX)
- lightweight AI/content guardrails
- export sanitization
- output validation
- citation validation
- optional persistent cache evaluation
- Phase 2 stabilization items

Deliverable:
- usable frontend with validated exports and basic guardrails

---

# PHASE 4 OBJECTIVE

Add production-safe observability to ContentBlitz using LangSmith tracing while preserving all existing architecture guarantees:

- deterministic LangGraph routing
- explicit state ownership
- stateless tools
- mockable providers
- no live external calls during normal tests
- no secrets in code, state, logs, docs, tests, or traces
- no frontend provider coupling
- no raw stack traces exposed to users

Phase 4 must be additive only. Do NOT rewrite graph architecture, agent responsibilities, provider tools, or UI orchestration behavior.

## PHASE 5+: TRUST + SAFETY SYSTEMS

Goal: Improve trustworthiness, observability, governance, and advanced safety

Implement:

- advanced prompt injection defense (Guardrails.ai)
- AI safety/moderation layer
- multi-stage validation/review pipelines
- AI judge/review systems
- optional human review workflows
- advanced trust scoring and citation analysis

Deliverable:
- enterprise-grade trust, safety, and governance infrastructure

---

## POST-PHASE STABILIZATION

Planned stabilization work after Phase 2 completion:

- improve clarification flow coverage
- improve provider edge-case coverage
- improve cache TTL/delete/serialization coverage
- review reducer merge edge cases
- evaluate persistent/distributed cache backend

---

# FINAL GOAL

The ContentBlitz system must be implemented as a deterministic, state-safe, test-driven LangGraph application.

It must satisfy all of the following:

- every node has explicit read/write state boundaries
- every routing decision is deterministic
- every retry path follows `RETRY_POLICY`
- every tool remains stateless
- every agent updates only its declared state fields
- every implementation step is validated by pytest
- no architectural decisions are left to Codex during implementation
- all behavior complies with the Consistency Enforcement Rules

## Consistency Enforcement Rules

- All retry logic MUST follow RETRY_POLICY
- All counters MUST be updated only by agent nodes, never by tools
- All routing decisions MUST be deterministic
- No function may rely on implicit behavior from another function
- All state mutations MUST be explicitly defined in agent contracts
- No node may mutate a state field unless that field appears in its "Writes to state" list
- No node may read a state field unless that field appears in its "Reads from state" list