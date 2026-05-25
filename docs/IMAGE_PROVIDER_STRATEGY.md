# Image Provider Strategy

## Scope

This document describes the current image-provider fallback strategy and safety behavior.

Primary implementation:

- `contentblitz/tools/generate_image.py`
- `contentblitz/tools/image.py`
- `contentblitz/agents/image_agent.py`

## Provider Fallback Chain

Default chain:

1. `stability_ai` (primary)
2. `fal_ai` (fallback)

Default models:

- primary model: `stable-image-core`
- fallback model: `fal-ai/flux/schnell`

Values are sourced from `MODEL_FALLBACKS` and can be overridden by environment variables.

## Runtime Controls

Provider execution is gated by:

- `CONTENTBLITZ_ENABLE_LIVE_CALLS`

When disabled, image generation returns a structured degraded result with:

- `code=live_calls_disabled`
- safe, recoverable messaging
- no provider client initialization

## Environment Variables

- `CONTENTBLITZ_IMAGE_PROVIDER`
- `CONTENTBLITZ_IMAGE_PROVIDER_FALLBACK`
- `CONTENTBLITZ_IMAGE_MODEL_PRIMARY`
- `CONTENTBLITZ_IMAGE_MODEL_FALLBACK`
- `STABILITY_API_KEY`
- `FAL_API_KEY` (or `FAL_KEY`)

## Result Metadata

Image tool results include safe provider attempt metadata:

- `provider_attempts`
- `provider_call_count`
- `provider_call_count_by_provider`
- `provider_latency_by_provider_ms`
- `primary_provider`
- `fallback_provider`
- `fallback_provider_attempted`
- `fallback_provider_used`

This metadata is designed for UI/observability use without exposing secrets or raw payloads.

## Recoverable Degraded Behavior

If providers fail or return unusable payloads:

- tool returns normalized degraded result
- image agent emits recoverable warning/error state
- workflow can continue with partial-success behavior for non-image outputs

No stack traces, API keys, or raw provider internals are exposed in UI-facing state.

## Cost-Control and Cache Interaction

- Image generation count is agent-owned (`image_generations_used_this_session`).
- Provider-returned counters do not own session counters.
- Research cache behavior is independent from image provider fallback.

## Mocking and Regression Testing

Provider fallback chain is intentionally mock-friendly:

- patch primary provider client builder to fail
- patch fallback provider client builder to succeed
- assert fallback metadata and recoverable behavior

Coverage references:

- `tests/unit/test_generate_image.py`
- `tests/integration/test_provider_contracts.py`
- `tests/integration/test_phase4_performance_contracts.py`
