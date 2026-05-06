"""Web search tool interface scaffold."""

from __future__ import annotations

from typing import Any, Dict, List


def search_web(query: str, depth: str = "standard") -> Dict[str, Any]:
    """Return an empty deterministic payload without external calls."""
    return {
        "query": query,
        "depth": depth,
        "provider_primary": "serp_api",
        "provider_fallback": "perplexity",
        "results": [],
        "used_external_api": False,
    }

