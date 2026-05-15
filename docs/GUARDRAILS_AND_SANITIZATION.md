# Guardrails and Sanitization

## Scope

This document describes implemented lightweight, deterministic safety controls in ContentBlitz.

These protections are rule-based and deterministic. They are not a full moderation/classification platform.

## Prompt Injection Handling

Primary files:

- `contentblitz/safety/prompt_injection.py`
- `contentblitz/agents/query_handler.py`

Behavior:

- Detects obvious prompt-injection patterns (case-insensitive, punctuation-tolerant).
- Emits safe metadata:
  - `prompt_injection_detected`
  - `prompt_injection_signals`
  - `sanitized_user_query`
- Sanitizes/neutralizes unsafe fragments before downstream prompt construction.
- Pure unsafe prompts route to safe clarification behavior.

## Output Sanitization

Primary file:

- `contentblitz/safety/output_sanitizer.py`

Applied across UI/export/persistence boundaries via renderer and serializer modules.

Sanitization removes or neutralizes unsafe patterns including:

- `<script>` blocks
- inline JS handlers (`onclick=`, `onerror=`, etc.)
- unsafe schemes (`javascript:`, `data:`, `file:`, `ftp:`, `mailto:`, `vbscript:`)
- unsafe embed tags (`iframe`, `object`, `embed`)
- stack trace markers
- raw provider payload/config markers
- API key/env-var leakage
- base64/data-image payloads

Safe `http`/`https` links are preserved.

## Citation Validation

Primary file:

- `contentblitz/quality/citations.py`

Behavior:

- validates source entries for missing/unsafe/duplicate fields
- rejects unsafe URL schemes
- avoids fake citation URLs
- returns deterministic degraded warning metadata when sources are weak

## Export Validation

Primary file:

- `contentblitz/tools/exports/validation.py`

Behavior:

- validates format structure and unsafe content patterns before marking export complete
- fails unsafe format outputs safely
- preserves valid outputs from other requested formats
- emits safe validation warnings/errors only

## Persistence Safety

Primary file:

- `contentblitz/persistence/serialization.py`

Behavior:

- stores sanitized content only
- strips unsafe progress/export/citation fields
- normalizes image error payloads
- avoids raw provider exceptions/payload dumps in persisted runs
