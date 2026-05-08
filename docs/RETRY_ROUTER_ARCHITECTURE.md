# Retry Router Architecture

## Overview

The Retry Router is responsible for controlled retry orchestration within the ContentBlitz workflow.

Its purpose is to:

- prevent infinite retry loops
- isolate retry ownership
- maintain deterministic retry behavior
- preserve workflow stability

---

# Retry Ownership Rules

## Critical Rule

Retry counters are owned exclusively by:

```text
retry_router_node
```

Writer agents must never increment retry counters directly.

---

# Managed State

The retry router manages:

```text
retry_counts
retry_feedback
cost_controls.total_retries_used_this_session
```

---

# Retry Flow

## Standard Flow

```text
quality_validator
    ↓
retry_router
    ↓
writer retry OR output assembly
```

---

# Retry Decision Logic

The retry router evaluates:

- validation status
- quality scores
- retry caps
- session retry limits

---

# Retry Limits

## Per-Agent Retry Limits

Retry limits prevent repeated regeneration loops.

## Session Retry Limits

Global retry limits prevent runaway orchestration behavior.

---

# Retry Feedback

Retry feedback is passed back to writers to guide improved regeneration.

Examples:

- clarity improvements
- structure improvements
- usefulness improvements

---

# Safety Guarantees

The retry system guarantees:

- no infinite loops
- deterministic retry routing
- bounded retry behavior
- safe workflow termination

---

# Forced Retry Testing

Retry behavior is validated using:

```text
tests/integration/test_retry_router_scenarios.py
```

These tests validate:

- single retries
- multi-output retries
- retry exhaustion
- retry cap enforcement
- retry routing correctness

---

# Known Limitations

Current retry behavior is deterministic and rule-based.

Future improvements may include:

- adaptive retry scoring
- semantic retry feedback
- model-assisted retry routing
- retry prioritization