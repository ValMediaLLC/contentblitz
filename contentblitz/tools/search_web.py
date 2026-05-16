"""SERP-backed web search tool with normalized provider contract."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional
from urllib import parse, request
from urllib.error import HTTPError, URLError

from contentblitz.tools.perplexity import search_perplexity
from contentblitz.tools.provider_types import SearchResult, SearchWebResult

_SERP_PROVIDER = "serp"
_PERPLEXITY_PROVIDER = "perplexity"
_AUTO_PROVIDER = "auto"
_SERP_ENDPOINT = "https://serpapi.com/search.json"
_DEFAULT_TIMEOUT_SECONDS = 15
_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_CAP = 20


def _safe_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _sanitize_query(query: str) -> str:
    return " ".join(str(query or "").strip().split())


def _safe_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        return candidate
    return None


def _credibility_score(url: Optional[str], source: str) -> float:
    if isinstance(url, str) and url:
        normalized = url.lower()
        if ".gov" in normalized or ".edu" in normalized:
            return 0.95
        if "wikipedia.org" in normalized:
            return 0.70
        return 0.80
    if str(source).strip().lower() == _PERPLEXITY_PROVIDER:
        return 0.35
    return 0.45


def _normalize_error(
    *,
    code: str,
    message: str,
    provider: str,
    recoverable: bool,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "provider": provider,
        "recoverable": recoverable,
        "status_code": status_code,
    }


def _degraded_result(
    *,
    provider: str,
    query: str,
    code: str,
    message: str,
    recoverable: bool,
    status_code: Optional[int] = None,
) -> SearchWebResult:
    return SearchWebResult(
        provider=provider,
        query=query,
        results=[],
        degraded=True,
        error=_normalize_error(
            code=code,
            message=message,
            provider=provider,
            recoverable=recoverable,
            status_code=status_code,
        ),
    )


def _build_serp_url(query: str, *, api_key: str, max_results: int) -> str:
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": max_results,
        "output": "json",
    }
    return f"{_SERP_ENDPOINT}?{parse.urlencode(params)}"


def _http_get_json(
    url: str, timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    req = request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ContentBlitz/phase2-search-web",
        },
        method="GET",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_serp_item(item: Mapping[str, Any]) -> SearchResult:
    title = str(item.get("title", "")).strip() or "Untitled result"
    url = _safe_url(item.get("link") or item.get("url"))
    snippet = (
        str(item.get("snippet", "")).strip() or str(item.get("description", "")).strip()
    )
    source = str(item.get("source", "")).strip() or "serp"
    published_at_raw = item.get("date")
    published_at = (
        str(published_at_raw).strip()
        if isinstance(published_at_raw, str) and published_at_raw.strip()
        else None
    )
    citation_available = bool(url)
    credibility = _credibility_score(url, source)
    return SearchResult(
        title=title,
        url=url,
        snippet=snippet,
        source=source,
        published_at=published_at,
        citation_available=citation_available,
        credibility_score=credibility,
    )


def _dedupe_exact_urls(results: List[SearchResult]) -> List[SearchResult]:
    seen_urls: set[str] = set()
    deduped: list[SearchResult] = []
    for result in results:
        if result.url is None:
            deduped.append(result)
            continue
        key = result.url.strip()
        if key in seen_urls:
            continue
        seen_urls.add(key)
        deduped.append(result)
    return deduped


def _search_serp(query: str, *, max_results: int) -> SearchWebResult:
    api_key = str(os.getenv("SERP_API_KEY", "")).strip()
    if not api_key:
        return _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="configuration_error",
            message="SERP_API_KEY is not configured.",
            recoverable=False,
        )

    url = _build_serp_url(query=query, api_key=api_key, max_results=max_results)
    try:
        payload = _http_get_json(url)
    except HTTPError as exc:
        return _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_http_error",
            message="SERP provider returned an HTTP error.",
            recoverable=True,
            status_code=exc.code if isinstance(exc.code, int) else None,
        )
    except URLError:
        return _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_unavailable",
            message="SERP provider is temporarily unavailable.",
            recoverable=True,
        )
    except json.JSONDecodeError:
        return _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_payload_invalid",
            message="SERP provider returned malformed JSON.",
            recoverable=True,
        )
    except Exception:
        return _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_error",
            message="SERP provider request failed.",
            recoverable=True,
        )

    organic_results = payload.get("organic_results")
    if not isinstance(organic_results, list):
        return _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_payload_invalid",
            message="SERP provider returned an unexpected payload shape.",
            recoverable=True,
        )

    normalized: list[SearchResult] = []
    for item in organic_results:
        if not isinstance(item, Mapping):
            continue
        normalized.append(_normalize_serp_item(item))

    deduped = _dedupe_exact_urls(normalized)
    deduped = deduped[:max_results]

    if not deduped and organic_results:
        return _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_payload_unusable",
            message="SERP provider payload did not contain usable results.",
            recoverable=True,
        )

    return SearchWebResult(
        provider=_SERP_PROVIDER,
        query=query,
        results=deduped,
        degraded=False,
        error=None,
    )


def _search_perplexity_placeholder(query: str, *, max_results: int) -> SearchWebResult:
    return search_perplexity(query=query, max_results=max_results)


def _is_unusable(result: SearchWebResult) -> bool:
    if result.degraded:
        return True
    if not result.results:
        return True
    for item in result.results:
        if str(item.snippet).strip():
            return False
    return True


def _search_auto(query: str, *, max_results: int) -> SearchWebResult:
    try:
        serp_result = _search_serp(query=query, max_results=max_results)
    except Exception:
        serp_result = _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_error",
            message="SERP provider request failed.",
            recoverable=True,
        )
    if not _is_unusable(serp_result):
        return serp_result

    try:
        perplexity_result = _search_perplexity_placeholder(
            query=query, max_results=max_results
        )
    except Exception:
        perplexity_result = _degraded_result(
            provider=_PERPLEXITY_PROVIDER,
            query=query,
            code="provider_error",
            message="Perplexity provider request failed.",
            recoverable=True,
        )
    if not _is_unusable(perplexity_result):
        return perplexity_result

    return SearchWebResult(
        provider=_AUTO_PROVIDER,
        query=query,
        results=[],
        degraded=True,
        error={
            "code": "all_providers_failed",
            "message": "SERP and Perplexity providers failed or returned unusable results.",
            "provider": _AUTO_PROVIDER,
            "recoverable": True,
            "providers_attempted": [_SERP_PROVIDER, _PERPLEXITY_PROVIDER],
            "serp_error": serp_result.error,
            "perplexity_error": perplexity_result.error,
        },
    )


def search_web(
    query: str,
    *,
    max_results: int = _DEFAULT_MAX_RESULTS,
    provider: str = _AUTO_PROVIDER,
) -> SearchWebResult:
    """Run a provider-backed web search and return normalized results."""
    safe_query = _sanitize_query(query)
    provider_name = str(provider or _SERP_PROVIDER).strip().lower()
    bounded_max_results = _safe_int(max_results, _DEFAULT_MAX_RESULTS)
    if bounded_max_results <= 0:
        bounded_max_results = _DEFAULT_MAX_RESULTS
    if bounded_max_results > _MAX_RESULTS_CAP:
        bounded_max_results = _MAX_RESULTS_CAP

    if not safe_query:
        return _degraded_result(
            provider=provider_name,
            query=safe_query,
            code="invalid_query",
            message="Search query is empty.",
            recoverable=False,
        )

    if provider_name == _SERP_PROVIDER:
        return _search_serp(safe_query, max_results=bounded_max_results)
    if provider_name == _PERPLEXITY_PROVIDER:
        return _search_perplexity_placeholder(
            safe_query, max_results=bounded_max_results
        )
    if provider_name == _AUTO_PROVIDER:
        return _search_auto(safe_query, max_results=bounded_max_results)

    return _degraded_result(
        provider=provider_name,
        query=safe_query,
        code="invalid_provider",
        message="Unsupported search provider.",
        recoverable=False,
    )


__all__ = ["search_web", "SearchResult", "SearchWebResult"]
