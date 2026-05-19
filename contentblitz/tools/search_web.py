"""SERP-backed web search tool with normalized provider contract."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional
from urllib import parse, request
from urllib.error import HTTPError, URLError

from contentblitz.config import live_provider_calls_enabled
from contentblitz.core.observability import safe_tool_metadata, start_tool_span
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


def _finish_tool_span(
    *,
    span: Any,
    started_at: float,
    span_name: str,
    provider: str,
    fallback_used: bool,
    fallback_provider: str = "",
    fallback_reason: str = "",
    retry_attempt: int = 0,
    retry_exhausted: bool = False,
    result: SearchWebResult | None = None,
    error: BaseException | None = None,
) -> None:
    duration_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    metadata: Dict[str, Any] = {
        "tool_name": span_name,
        "provider": provider,
        "fallback_used": fallback_used,
        "fallback_provider": str(fallback_provider or "").strip(),
        "fallback_reason": str(fallback_reason or "").strip(),
        "retry_attempt": max(0, int(retry_attempt)),
        "retry_exhausted": bool(retry_exhausted),
        "duration_ms": duration_ms,
    }
    outputs: Dict[str, Any] = {}

    if result is not None:
        citation_count = sum(1 for item in result.results if item.citation_available)
        metadata.update(
            {
                "provider": result.provider or provider,
                "degraded": result.degraded,
                "result_count": len(result.results),
                "source_count": len(result.results),
                "citation_available_count": citation_count,
            }
        )
        outputs = {
            "provider": result.provider,
            "degraded": result.degraded,
            "result_count": len(result.results),
        }
    elif error is not None:
        metadata["degraded"] = True
        if not metadata["fallback_reason"]:
            metadata["fallback_reason"] = "tool_exception"

    span.finish(
        metadata=safe_tool_metadata(metadata),
        outputs=outputs,
        error=error,
    )


def _run_provider_span(
    *,
    span_name: str,
    provider: str,
    fallback_used: bool,
    fallback_reason: str = "",
    query: str,
    max_results: int,
    search_fn: Any,
) -> SearchWebResult:
    started_at = time.perf_counter()
    provider_span = start_tool_span(
        span_name,
        metadata={
            "provider": provider,
            "fallback_used": fallback_used,
            "fallback_provider": provider if fallback_used else "",
            "fallback_reason": fallback_reason,
        },
        inputs={"tool_name": span_name, "provider": provider},
    )
    try:
        result = search_fn(query=query, max_results=max_results)
    except Exception as error:
        _finish_tool_span(
            span=provider_span,
            started_at=started_at,
            span_name=span_name,
            provider=provider,
            fallback_used=fallback_used,
            fallback_provider=provider if fallback_used else "",
            fallback_reason=fallback_reason,
            retry_attempt=1,
            retry_exhausted=False,
            error=error,
        )
        raise
    _finish_tool_span(
        span=provider_span,
        started_at=started_at,
        span_name=span_name,
        provider=provider,
        fallback_used=fallback_used,
        fallback_provider=provider if fallback_used else "",
        fallback_reason=fallback_reason,
        retry_attempt=1,
        retry_exhausted=False,
        result=result,
    )
    return result


def _fallback_reason_from_result(result: SearchWebResult) -> str:
    if not result.degraded:
        return "unusable_results"
    if isinstance(result.error, Mapping):
        code = str(result.error.get("code", "")).strip()
        if code:
            return code
    return "provider_error"


def _search_auto(query: str, *, max_results: int) -> tuple[SearchWebResult, bool]:
    try:
        serp_result = _run_provider_span(
            span_name="serp",
            provider=_SERP_PROVIDER,
            fallback_used=False,
            fallback_reason="",
            query=query,
            max_results=max_results,
            search_fn=_search_serp,
        )
    except Exception:
        serp_result = _degraded_result(
            provider=_SERP_PROVIDER,
            query=query,
            code="provider_error",
            message="SERP provider request failed.",
            recoverable=True,
        )
    if not _is_unusable(serp_result):
        return serp_result, False

    try:
        fallback_reason = _fallback_reason_from_result(serp_result)
        perplexity_result = _run_provider_span(
            span_name="perplexity_fallback",
            provider=_PERPLEXITY_PROVIDER,
            fallback_used=True,
            fallback_reason=fallback_reason,
            query=query,
            max_results=max_results,
            search_fn=_search_perplexity_placeholder,
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
        return perplexity_result, True

    return (
        SearchWebResult(
            provider=_AUTO_PROVIDER,
            query=query,
            results=[],
            degraded=True,
            error={
                "code": "all_providers_failed",
                "message": (
                    "SERP and Perplexity providers failed or returned "
                    "unusable results."
                ),
                "provider": _AUTO_PROVIDER,
                "recoverable": True,
                "providers_attempted": [_SERP_PROVIDER, _PERPLEXITY_PROVIDER],
                "serp_error": serp_result.error,
                "perplexity_error": perplexity_result.error,
            },
        ),
        True,
    )


def search_web(
    query: str,
    *,
    max_results: int = _DEFAULT_MAX_RESULTS,
    provider: str = _AUTO_PROVIDER,
) -> SearchWebResult:
    """Run a provider-backed web search and return normalized results."""
    started_at = time.perf_counter()
    safe_query = _sanitize_query(query)
    provider_name = str(provider or _SERP_PROVIDER).strip().lower()
    tool_span = start_tool_span(
        "search_web",
        metadata={"provider": provider_name},
        inputs={"tool_name": "search_web", "provider": provider_name},
    )
    bounded_max_results = _safe_int(max_results, _DEFAULT_MAX_RESULTS)
    if bounded_max_results <= 0:
        bounded_max_results = _DEFAULT_MAX_RESULTS
    if bounded_max_results > _MAX_RESULTS_CAP:
        bounded_max_results = _MAX_RESULTS_CAP

    def _finalize(result: SearchWebResult, *, fallback_used: bool) -> SearchWebResult:
        fallback_reason = ""
        fallback_provider = ""
        retry_attempt = 1
        if fallback_used:
            fallback_provider = result.provider or _PERPLEXITY_PROVIDER
            fallback_reason = "provider_fallback"
            retry_attempt = 2
        if result.degraded and isinstance(result.error, Mapping):
            code = str(result.error.get("code", "")).strip()
            if code:
                fallback_reason = code
        _finish_tool_span(
            span=tool_span,
            started_at=started_at,
            span_name="search_web",
            provider=provider_name,
            fallback_used=fallback_used,
            fallback_provider=fallback_provider,
            fallback_reason=fallback_reason,
            retry_attempt=retry_attempt,
            retry_exhausted=result.degraded,
            result=result,
        )
        return result

    if not safe_query:
        return _finalize(
            _degraded_result(
                provider=provider_name,
                query=safe_query,
                code="invalid_query",
                message="Search query is empty.",
                recoverable=False,
            ),
            fallback_used=False,
        )

    if not live_provider_calls_enabled():
        return _finalize(
            _degraded_result(
                provider=provider_name,
                query=safe_query,
                code="live_calls_disabled",
                message=(
                    "Live provider calls are disabled by "
                    "CONTENTBLITZ_ENABLE_LIVE_CALLS."
                ),
                recoverable=False,
            ),
            fallback_used=False,
        )

    try:
        if provider_name == _SERP_PROVIDER:
            result = _run_provider_span(
                span_name="serp",
                provider=_SERP_PROVIDER,
                fallback_used=False,
                fallback_reason="",
                query=safe_query,
                max_results=bounded_max_results,
                search_fn=_search_serp,
            )
            return _finalize(result, fallback_used=False)
        if provider_name == _PERPLEXITY_PROVIDER:
            result = _run_provider_span(
                span_name="perplexity",
                provider=_PERPLEXITY_PROVIDER,
                fallback_used=False,
                fallback_reason="",
                query=safe_query,
                max_results=bounded_max_results,
                search_fn=_search_perplexity_placeholder,
            )
            return _finalize(result, fallback_used=False)
        if provider_name == _AUTO_PROVIDER:
            result, fallback_used = _search_auto(
                safe_query, max_results=bounded_max_results
            )
            return _finalize(result, fallback_used=fallback_used)

        return _finalize(
            _degraded_result(
                provider=provider_name,
                query=safe_query,
                code="invalid_provider",
                message="Unsupported search provider.",
                recoverable=False,
            ),
            fallback_used=False,
        )
    except Exception as error:
        _finish_tool_span(
            span=tool_span,
            started_at=started_at,
            span_name="search_web",
            provider=provider_name,
            fallback_used=False,
            fallback_provider="",
            fallback_reason="tool_exception",
            retry_attempt=1,
            retry_exhausted=False,
            error=error,
        )
        raise


__all__ = ["search_web", "SearchResult", "SearchWebResult"]
