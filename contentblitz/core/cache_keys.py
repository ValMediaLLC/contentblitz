"""Cache key helpers for deterministic research caching."""

from __future__ import annotations

import hashlib


def normalize_query(query: str) -> str:
    """Normalize query text before hashing."""
    return " ".join(str(query or "").strip().lower().split())


def sha256_normalized_query(query: str) -> str:
    """Return SHA256 hex digest of the normalized query."""
    normalized = normalize_query(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_research_cache_key(query: str, depth: str = "standard") -> str:
    """
    Build a deterministic research cache key:
    research:{sha256_normalized_query}:{depth}
    """
    normalized_depth = str(depth or "standard").strip().lower() or "standard"
    return f"research:{sha256_normalized_query(query)}:{normalized_depth}"


__all__ = [
    "normalize_query",
    "sha256_normalized_query",
    "build_research_cache_key",
]
