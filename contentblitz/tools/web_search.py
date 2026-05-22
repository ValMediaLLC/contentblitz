"""Compatibility adapter for legacy web search call sites."""

from __future__ import annotations

from typing import Any, Dict

from contentblitz.tools.search_web import search_web as _core_search_web


def _legacy_result_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": item.get("title"),
        "url": item.get("url"),
        "snippet": item.get("snippet"),
        "source": item.get("source"),
        "date": item.get("published_at"),
        "citation_available": bool(item.get("citation_available", False)),
        "credibility_score": float(item.get("credibility_score", 0.0)),
    }


def search_web(
    query: str,
    depth: str = "standard",
    timeout_seconds: float | None = None,
) -> Dict[str, Any]:
    """
    Legacy dict contract used by the research agent.

    - `depth="standard"` -> SERP provider
    - `depth="fallback"` -> Perplexity placeholder provider
    """
    provider = "perplexity" if str(depth).strip().lower() == "fallback" else "serp"
    typed_result = _core_search_web(
        query=query,
        max_results=5,
        provider=provider,
        timeout_seconds=timeout_seconds,
    )
    return {
        "query": typed_result.query,
        "depth": depth,
        "provider_primary": "serp_api",
        "provider_fallback": "perplexity",
        "provider_used": typed_result.provider,
        "results": [
            _legacy_result_item(item.as_dict()) for item in typed_result.results
        ],
        "used_external_api": not typed_result.degraded,
        "degraded": typed_result.degraded,
        "error": typed_result.error,
    }
