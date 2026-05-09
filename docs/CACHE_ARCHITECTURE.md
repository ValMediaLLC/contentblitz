# Cache Architecture

## Scope

Phase 2 research caching is implemented in:

- `contentblitz/tools/cache.py`
- `contentblitz/core/cache_keys.py`
- `contentblitz/agents/research_agent.py`

## Cache Backend

- Default backend: in-memory process store
- Storage location: module-level `_CACHE_STORE` in `tools/cache.py`
- Persistence: process-local only (not shared across process restarts)

## Cache Key Rule

Canonical key format:

```text
research:{sha256_normalized_query}:{depth}
```

Key generation rules:

- Query is normalized (`strip`, `lower`, collapse whitespace)
- SHA256 hash is used
- Raw query text is not included in key output
- Exact match only

## TTL Behavior

- Default TTL: `1800` seconds
- TTL is read from `state["cache_metadata"]["ttl_seconds"]` when writing
- Expired entries are treated as misses and removed on read

## Read/Write Flow

Research agent flow:

1. Build research cache key from query + depth
2. If cache enabled and key exists and is unexpired:
   - return cached payload
   - skip provider search calls
   - do not increment search query counters
3. On miss:
   - perform provider-backed research path
   - cache only successful non-degraded research payloads
4. Degraded/provider-error results are not cached

## Cached Payload Shape

Cached research payload includes:

- `research_data` (sanitized/normalized)
- `sources` (sanitized)

Excluded:

- secrets
- raw provider exceptions
- raw user input inside cache key

## Cache Metadata in State

State metadata tracks:

- `enabled`
- `ttl_seconds`
- `backend`
- `keys`

Important:

- cache tool functions return state patch data
- cache tool does not mutate state directly
- agent applies metadata updates

## Operational Notes

- Cache can be disabled via `state["cache_metadata"]["enabled"] = False`
- `clear_cache()` exists for test isolation and manual validation
- Values must be JSON-serializable to be cached

## Security Rules

- `.env` is never committed.
- API keys are read only from environment variables.
- Tools are stateless.
- State never stores secrets.
- Provider errors are normalized.
- Base64 image data is never stored in state.
