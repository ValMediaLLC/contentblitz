"""Perplexity search provider integration for web-search fallback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional
from urllib import request
from urllib.error import HTTPError, URLError

from contentblitz.config import live_provider_calls_enabled
from contentblitz.tools.provider_types import SearchResult, SearchWebResult

_PROVIDER = "perplexity"
_API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
_DEFAULT_TIMEOUT_SECONDS = 20


def _safe_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        return candidate
    return None


def _credibility_score(url: Optional[str]) -> float:
    if isinstance(url, str) and url:
        return 0.80
    return 0.35


def _normalize_error(
    *,
    code: str,
    message: str,
    recoverable: bool,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "provider": _PROVIDER,
        "recoverable": recoverable,
        "status_code": status_code,
    }


def _degraded_result(
    *,
    query: str,
    code: str,
    message: str,
    recoverable: bool,
    status_code: Optional[int] = None,
) -> SearchWebResult:
    return SearchWebResult(
        provider=_PROVIDER,
        query=query,
        results=[],
        degraded=True,
        error=_normalize_error(
            code=code,
            message=message,
            recoverable=recoverable,
            status_code=status_code,
        ),
    )


def _http_post_json(
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    raw_payload = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        headers=headers,
        data=raw_payload,
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _extract_answer_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    output = payload.get("output")
    if isinstance(output, str) and output.strip():
        return output.strip()
    return ""


def _extract_citation_urls(payload: Mapping[str, Any]) -> List[str]:
    urls: list[str] = []

    def _collect_from(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    for key in ("url", "link"):
                        candidate = _safe_url(item.get(key))
                        if candidate:
                            urls.append(candidate)
                elif isinstance(item, str):
                    candidate = _safe_url(item)
                    if candidate:
                        urls.append(candidate)

    _collect_from(payload.get("citations"))

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            _collect_from(first.get("citations"))
            message = first.get("message")
            if isinstance(message, Mapping):
                _collect_from(message.get("citations"))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def search_perplexity(query: str, *, max_results: int = 5) -> SearchWebResult:
    if not live_provider_calls_enabled():
        return _degraded_result(
            query=query,
            code="live_calls_disabled",
            message="Live provider calls are disabled by CONTENTBLITZ_ENABLE_LIVE_CALLS.",
            recoverable=False,
        )

    api_key = str(os.getenv("PERPLEXITY_API_KEY", "")).strip()
    if not api_key:
        return _degraded_result(
            query=query,
            code="configuration_error",
            message="PERPLEXITY_API_KEY is not configured.",
            recoverable=False,
        )

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "Provide concise factual web-backed information with citations when available.",
            },
            {"role": "user", "content": query},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ContentBlitz/phase2-perplexity-fallback",
    }

    try:
        response = _http_post_json(url=_API_ENDPOINT, payload=payload, headers=headers)
    except HTTPError as exc:
        return _degraded_result(
            query=query,
            code="provider_http_error",
            message="Perplexity provider returned an HTTP error.",
            recoverable=True,
            status_code=exc.code if isinstance(exc.code, int) else None,
        )
    except URLError:
        return _degraded_result(
            query=query,
            code="provider_unavailable",
            message="Perplexity provider is temporarily unavailable.",
            recoverable=True,
        )
    except json.JSONDecodeError:
        return _degraded_result(
            query=query,
            code="provider_payload_invalid",
            message="Perplexity provider returned malformed JSON.",
            recoverable=True,
        )
    except Exception:
        return _degraded_result(
            query=query,
            code="provider_error",
            message="Perplexity provider request failed.",
            recoverable=True,
        )

    answer_text = _extract_answer_text(response)
    citation_urls = _extract_citation_urls(response)

    if not answer_text:
        return _degraded_result(
            query=query,
            code="provider_payload_unusable",
            message="Perplexity provider returned no usable answer text.",
            recoverable=True,
        )

    results: list[SearchResult] = []
    if citation_urls:
        for index, url in enumerate(citation_urls[:max_results], start=1):
            results.append(
                SearchResult(
                    title=f"Perplexity result {index}",
                    url=url,
                    snippet=answer_text,
                    source=_PROVIDER,
                    published_at=None,
                    citation_available=True,
                    credibility_score=_credibility_score(url),
                )
            )
    else:
        results.append(
            SearchResult(
                title="Perplexity answer",
                url=None,
                snippet=answer_text,
                source=_PROVIDER,
                published_at=None,
                citation_available=False,
                credibility_score=_credibility_score(None),
            )
        )

    return SearchWebResult(
        provider=_PROVIDER,
        query=query,
        results=results,
        degraded=False,
        error=None,
    )


__all__ = ["search_perplexity"]
