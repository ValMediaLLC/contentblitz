# ContentBlitz Testing Strategy

## Overview

ContentBlitz uses a layered testing strategy designed to validate:

- deterministic orchestration behavior
- graph routing correctness
- state ownership integrity
- regression protection
- retry safety
- recoverable failure handling

---

# Test Layers

## Unit Tests

Unit tests validate isolated agent and utility behavior.

Examples:

- query handler classification
- retry router decisions
- quality validator scoring
- export formatting
- clarification handling

Location:

```text
tests/unit/
```

---

## Integration Tests

Integration tests validate full orchestration flows and multi-agent interactions.

Examples:

- prompt regression flows
- retry routing
- output assembly
- export generation
- clarification routing
- error handling

Location:

```text
tests/integration/
```

---

## Prompt Regression Testing

Prompt regression tests ensure that:

- routing remains stable
- expected outputs are generated
- retry behavior remains deterministic
- fallback behavior remains safe

These tests act as release gates.

---

## Smoke Testing

Smoke testing validates end-to-end execution manually using representative prompts.

Script:

```bash
python scripts/dev/smoke_query_handler.py
```

---

# Retry Validation

Retry behavior is validated through:

- integration retry scenarios
- forced retry state testing
- retry counter validation
- retry cap enforcement

---

# Failure Handling Validation

Failure paths validated include:

- degraded research synthesis
- image generation failures
- clarification fallback
- fatal error handling
- retry exhaustion

---

# Coverage Goals

Target coverage:

```text
90%+
```

Critical orchestration/state modules should maintain high coverage.

Wrapper modules and compatibility layers may be excluded or lightly tested.

---

# Deterministic Testing Rules

Tests must:

- avoid real external APIs
- remain deterministic
- avoid flaky provider behavior
- avoid network dependencies

---

# Future Testing Areas

Planned Phase 2 additions:

- performance benchmarking
- concurrency testing
- persistence testing
- streaming validation
- provider integration testing
- load testing