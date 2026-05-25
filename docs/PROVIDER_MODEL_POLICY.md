# Provider Model Policy

## Scope

This document describes text provider/model selection policy used by ContentBlitz agent nodes.

Primary implementation:

- `contentblitz/core/model_policy.py`
- `contentblitz/core/cost_controls.py`
- `contentblitz/tools/text.py`

## Defaults

Default provider policy is OpenAI-backed unless overridden:

- default provider: `openai`
- default high-quality model: `gpt-4o`
- default low-cost fallback model: `gpt-4o-mini`

Agent entries include:

- `query_handler`
- `research_agent`
- `content_strategist`
- `blog_writer`
- `linkedin_writer`
- `image_agent` (for prompt text enhancement)
- `quality_validator`
- `retry_rewrite`
- `clarification`
- plus `default` and `unknown` safety entries

## Resolution Rules

Policy resolution uses deterministic precedence:

1. Explicit per-agent env override (for example `CONTENTBLITZ_TEXT_MODEL_RESEARCH_AGENT_DEFAULT`)
2. JSON policy override in `CONTENTBLITZ_AGENT_MODEL_POLICY`
3. Global model overrides (`CONTENTBLITZ_TEXT_MODEL_DEFAULT`, `CONTENTBLITZ_TEXT_MODEL_FALLBACK`, alias `CONTENTBLITZ_DEFAULT_TEXT_MODEL`)
4. Built-in policy defaults

Provider override:

- `CONTENTBLITZ_TEXT_PROVIDER` can switch global provider to `anthropic`
- fallback provider defaults to the same provider unless explicitly overridden

## Budget-Aware Selection

`preferred_text_model(...)` applies near-budget routing using cost controls:

- when near budget threshold is reached, fallback model is selected
- threshold is computed by `near_token_budget(...)`
- this selection is still policy-driven and deterministic

## Model Policy and Retry/Cost Controls

- Model choice does not bypass retry policy.
- Retry limits remain governed by `RETRY_POLICY` and agent orchestration.
- Token accounting remains agent-owned (`apply_text_tokens`) and independent from provider-side billing semantics.

## Environment Variables

Common variables:

- `CONTENTBLITZ_TEXT_PROVIDER`
- `CONTENTBLITZ_TEXT_MODEL_DEFAULT`
- `CONTENTBLITZ_TEXT_MODEL_FALLBACK`
- `CONTENTBLITZ_DEFAULT_TEXT_MODEL`
- `CONTENTBLITZ_AGENT_MODEL_POLICY`
- `CONTENTBLITZ_TEXT_MODEL_<AGENT>_DEFAULT`
- `CONTENTBLITZ_TEXT_MODEL_<AGENT>_FALLBACK`

Example:

```env
CONTENTBLITZ_TEXT_PROVIDER=anthropic
CONTENTBLITZ_TEXT_MODEL_RESEARCH_AGENT_DEFAULT=claude-sonnet-4-6
CONTENTBLITZ_TEXT_MODEL_RESEARCH_AGENT_FALLBACK=claude-haiku-4-5-20251001
```

## Test Coverage

Model policy behavior is validated by:

- `tests/unit/test_model_policy.py`
- `tests/integration/test_phase5_performance_contracts.py`

These tests verify deterministic selection and agent-side policy usage without requiring live provider calls.
