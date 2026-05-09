# Provider Contracts

## Purpose

This document defines the normalized contracts returned by Phase 2 provider tools.

## Text Generation Contract

Tool:

- `contentblitz/tools/generate_text.py`

Signature:

```python
generate_text(
    prompt: str,
    agent_key: str,
    model: str = "gpt-4o",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerateTextResult
```

`GenerateTextResult` fields:

- `text`
- `model`
- `provider`
- `input_tokens`
- `output_tokens`
- `total_tokens`
- `degraded`
- `error`

Behavior:

- Primary model: `gpt-4o`
- Fallback model: `gpt-4o-mini`
- Retries per model: `RETRY_POLICY[agent_key] + 1`
- Invalid `agent_key` fails safely (`degraded=True`)

## Web Search Contract

Tool:

- `contentblitz/tools/search_web.py`

Signature:

```python
search_web(
    query: str,
    *,
    max_results: int = 5,
    provider: str = "auto",
) -> SearchWebResult
```

`SearchWebResult` fields:

- `provider`
- `query`
- `results`
- `degraded`
- `error`

Each result item fields:

- `title`
- `url`
- `snippet`
- `source`
- `published_at`
- `citation_available`
- `credibility_score`

Behavior:

- `provider="serp"` uses SERP only
- `provider="perplexity"` uses Perplexity only
- `provider="auto"` tries SERP first, then Perplexity if SERP degraded/unusable
- Exact duplicate URLs are removed inside a tool response

## Image Generation Contract

Tool:

- `contentblitz/tools/generate_image.py`

Signature:

```python
generate_image(
    prompt: str,
    *,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str | None = None,
) -> GenerateImageResult
```

`GenerateImageResult` fields:

- `provider`
- `model`
- `prompt`
- `image_url`
- `revised_prompt`
- `degraded`
- `error`

Behavior:

- Primary model: `dall-e-3`
- Fallback model: `dall-e-2`
- Returns URL or provider file reference only
- No base64 payloads are returned

## Error Contract Rules

Across all provider tools:

- Missing API key returns structured configuration error
- Provider/network failures return structured normalized error
- Raw provider exceptions are not returned directly
- Errors include safe metadata (`code`, `message`, `provider`, `recoverable`, and optional status details)

## State Ownership Contract

- Tools are stateless.
- Tools do not mutate ContentBlitz state.
- Tools do not update counters.
- Agent nodes apply all state updates from normalized tool outputs.

## Security Rules

- `.env` is never committed.
- API keys are read only from environment variables.
- Tools are stateless.
- State never stores secrets.
- Provider errors are normalized.
- Base64 image data is never stored in state.
