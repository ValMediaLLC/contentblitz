"""Shared provider result types for Phase 2 tool contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SearchResult:
    """Normalized search result item used across web-search providers."""

    title: str
    url: Optional[str]
    snippet: str
    source: str
    published_at: Optional[str]
    citation_available: bool
    credibility_score: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_at": self.published_at,
            "citation_available": self.citation_available,
            "credibility_score": self.credibility_score,
        }


@dataclass(frozen=True)
class SearchWebResult:
    """Normalized top-level web-search tool result."""

    provider: str
    query: str
    results: list[SearchResult]
    degraded: bool
    error: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "query": self.query,
            "results": [item.as_dict() for item in self.results],
            "degraded": self.degraded,
            "error": self.error,
        }


__all__ = ["SearchResult", "SearchWebResult"]
