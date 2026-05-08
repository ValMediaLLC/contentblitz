# Known Limitations

## Overview

This document tracks known limitations and intentional constraints within the current Phase 1 implementation.

---

# Deterministic Provider Behavior

Phase 1 intentionally avoids real external provider integrations.

The following systems currently use deterministic fallback behavior:

- research synthesis
- image generation
- clarification fallback
- export formatting

---

# Image Generation

Image generation currently does not return real generated image assets.

Current behavior:

- image prompts are generated
- recoverable failures are returned safely
- orchestration continues without crashing

Future phases will introduce:

- real provider integration
- placeholder asset support
- image persistence/storage

---

# Research Quality

Degraded research paths currently use deterministic fallback summaries.

This behavior prioritizes:

- stability
- deterministic testing
- recoverable orchestration behavior

---

# Query Classification

Some prompts may over-request outputs due to conservative routing logic.

Example:

- LinkedIn-oriented prompts may also trigger blog output generation.

Planned improvements:

- improved intent classification
- confidence scoring
- semantic routing refinement

---

# LangGraph Warning

Current LangGraph serializer configuration produces a deprecation warning related to:

```text
allowed_objects
```

This does not affect orchestration correctness but should be addressed in a future cleanup phase.

---

# Documentation Gaps

Some architecture and operational documentation remains incomplete.

Planned additions include:

- persistence architecture
- provider integration architecture
- deployment documentation
- performance benchmarking documentation

---

# No Persistence Layer

Phase 1 currently uses in-memory orchestration state.

No database or persistent storage layer is implemented yet.

---

# No Real Provider Integrations

Phase 1 intentionally avoids:

- real OpenAI calls
- real image provider calls
- external vector stores
- external databases

This constraint exists to maintain deterministic testing behavior.

---

# Future Improvements

Planned Phase 2 improvements include:

- real provider integrations
- persistence support
- advanced retries
- streaming outputs
- human review workflows
- performance optimization
- observability tooling
- analytics support