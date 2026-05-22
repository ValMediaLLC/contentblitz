"""Research agent node implementation for Phase 1/2 placeholder flow."""

from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Mapping
from urllib.parse import urlparse

from contentblitz.core.cost_controls import (
    apply_text_tokens,
    normalize_cost_controls,
    preferred_text_model,
    search_cap_reached,
    token_budget_exceeded,
)
from contentblitz.tools.cache import (
    build_research_cache_key,
    get_cached_research,
    set_cached_research,
    touch_cached_research_key,
)
from contentblitz.tools.text import generate_text
from contentblitz.tools.web_search import search_web

_MIN_SEARCH_QUERIES = 3
_MAX_SEARCH_QUERIES = 5
_DEFAULT_SEARCH_QUERY_CAP = 5
_SEARCH_FANOUT_CONCURRENCY = 5
_SEARCH_QUERY_TIMEOUT_SECONDS = 5.0
# SERP/search-provider fan-out wall timeout only (not full research node timeout).
_SEARCH_PROVIDER_WALL_TIMEOUT_SECONDS = 8.0
_MIN_LIST_ITEMS = 3
_FALLBACK_KEYWORDS = ["market trends", "audience insights", "strategic positioning"]
_KEYWORD_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "about",
    "how",
    "what",
    "why",
    "is",
    "are",
    "be",
    "as",
    "by",
    "from",
    "this",
    "that",
}
_KEYWORD_PRESERVE_TOKENS = {
    "linkedin",
    "blog",
    "image",
    "research",
    "not",
    "without",
    "vs",
    "ai",
    "seo",
    "ux",
}
_THEME_RULES = (
    (
        ("adoption", "demand", "growth", "market", "sales"),
        "market growth and adoption",
    ),
    (
        ("2026", "availability", "launch", "model", "release"),
        "model availability and release timelines",
    ),
    (
        ("affordability", "cost", "ownership", "price", "pricing"),
        "pricing and total cost of ownership",
    ),
    (
        ("charging", "infrastructure", "network", "station"),
        "charging infrastructure and accessibility",
    ),
    (
        ("battery", "efficiency", "performance", "range"),
        "range and battery performance",
    ),
)


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return None


def _provider_from_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if not normalized:
        return ""
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("gpt-") or normalized.startswith("o"):
        return "openai"
    return ""


def _provider_from_llm_response(
    response: Mapping[str, Any],
    *,
    fallback_model: str = "",
) -> str:
    provider = str(response.get("provider", "")).strip().lower()
    if provider:
        return provider
    model = str(response.get("model", "")).strip() or str(fallback_model).strip()
    inferred = _provider_from_model_name(model)
    if inferred:
        return inferred
    return "openai"


def _increment_int_map(
    target: Dict[str, int],
    *,
    key: str,
    delta: int = 1,
) -> None:
    safe_key = str(key).strip().lower()
    if not safe_key:
        return
    amount = max(0, int(delta))
    if amount <= 0:
        target.setdefault(safe_key, max(0, int(target.get(safe_key, 0))))
        return
    target[safe_key] = max(0, int(target.get(safe_key, 0))) + amount


@dataclass(frozen=True)
class _SearchCallResult:
    query: str
    depth: str
    response: Dict[str, Any]
    duration_ms: int
    attempted: bool
    timed_out: bool


def _node_time_remaining_seconds(deadline: float) -> float:
    return max(0.0, float(deadline - perf_counter()))


def _invoke_search_web(
    *,
    query: str,
    depth: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    try:
        return _safe_dict(
            search_web(query=query, depth=depth, timeout_seconds=timeout_seconds)
        )
    except TypeError as exc:
        if "timeout_seconds" not in str(exc):
            raise
        return _safe_dict(search_web(query=query, depth=depth))


async def _run_search_call_async(
    *,
    query: str,
    depth: str,
    semaphore: asyncio.Semaphore,
    per_query_timeout_seconds: float,
    node_deadline: float,
) -> _SearchCallResult:
    async with semaphore:
        remaining_seconds = _node_time_remaining_seconds(node_deadline)
        if remaining_seconds <= 0.0:
            return _SearchCallResult(
                query=query,
                depth=depth,
                response={"results": []},
                duration_ms=0,
                attempted=False,
                timed_out=True,
            )
        effective_timeout = min(per_query_timeout_seconds, remaining_seconds)
        started_at = perf_counter()
        timed_out = False
        attempted = True
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    _invoke_search_web,
                    query=query,
                    depth=depth,
                    timeout_seconds=effective_timeout,
                ),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            response = {"results": []}
            timed_out = True
        except Exception:
            response = {"results": []}
        duration_ms = max(0, int((perf_counter() - started_at) * 1000))
        return _SearchCallResult(
            query=query,
            depth=depth,
            response=_safe_dict(response),
            duration_ms=duration_ms,
            attempted=attempted,
            timed_out=timed_out,
        )


async def _run_search_fanout_async(
    *,
    queries: List[str],
    depth: str,
    max_concurrency: int,
    per_query_timeout_seconds: float,
    node_deadline: float,
) -> tuple[List[_SearchCallResult], bool]:
    if not queries:
        return [], False
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    task_by_query: dict[str, asyncio.Task[_SearchCallResult]] = {}
    for search_query in queries:
        task_by_query[search_query] = asyncio.create_task(
            _run_search_call_async(
                query=search_query,
                depth=depth,
                semaphore=semaphore,
                per_query_timeout_seconds=per_query_timeout_seconds,
                node_deadline=node_deadline,
            )
        )

    phase_timeout_seconds = _node_time_remaining_seconds(node_deadline)
    if phase_timeout_seconds <= 0.0:
        for task in task_by_query.values():
            task.cancel()
        return (
            [
                _SearchCallResult(
                    query=search_query,
                    depth=depth,
                    response={"results": []},
                    duration_ms=0,
                    attempted=False,
                    timed_out=True,
                )
                for search_query in queries
            ],
            True,
        )

    done, pending = await asyncio.wait(
        task_by_query.values(),
        timeout=phase_timeout_seconds,
    )
    wall_timeout_triggered = bool(pending)
    for task in pending:
        task.cancel()

    done_results: dict[str, _SearchCallResult] = {}
    for task in done:
        try:
            result = task.result()
        except Exception:
            continue
        done_results[result.query] = result

    ordered_results: list[_SearchCallResult] = []
    for search_query in queries:
        result = done_results.get(search_query)
        if result is not None:
            ordered_results.append(result)
            continue
        ordered_results.append(
            _SearchCallResult(
                query=search_query,
                depth=depth,
                response={"results": []},
                duration_ms=0,
                attempted=False,
                timed_out=True,
            )
        )
    return ordered_results, wall_timeout_triggered


def _run_search_fanout(
    *,
    queries: List[str],
    depth: str,
    max_concurrency: int,
    per_query_timeout_seconds: float,
    node_deadline: float,
) -> tuple[List[_SearchCallResult], bool]:
    if not queries:
        return [], False
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _run_search_fanout_async(
                queries=queries,
                depth=depth,
                max_concurrency=max_concurrency,
                per_query_timeout_seconds=per_query_timeout_seconds,
                node_deadline=node_deadline,
            )
        )

    ordered_results: list[_SearchCallResult] = []
    wall_timeout_triggered = False
    for search_query in queries:
        remaining_seconds = _node_time_remaining_seconds(node_deadline)
        if remaining_seconds <= 0.0:
            wall_timeout_triggered = True
            ordered_results.append(
                _SearchCallResult(
                    query=search_query,
                    depth=depth,
                    response={"results": []},
                    duration_ms=0,
                    attempted=False,
                    timed_out=True,
                )
            )
            continue

        effective_timeout = min(per_query_timeout_seconds, remaining_seconds)
        started_at = perf_counter()
        timed_out = False
        try:
            response = _invoke_search_web(
                query=search_query,
                depth=depth,
                timeout_seconds=effective_timeout,
            )
        except Exception:
            response = {"results": []}
        duration_ms = max(0, int((perf_counter() - started_at) * 1000))
        error = _safe_dict(_safe_dict(response).get("error", {}))
        error_code = str(error.get("code", "")).strip().lower()
        if error_code == "provider_timeout":
            timed_out = True
        ordered_results.append(
            _SearchCallResult(
                query=search_query,
                depth=depth,
                response=_safe_dict(response),
                duration_ms=duration_ms,
                attempted=True,
                timed_out=timed_out,
            )
        )

    return ordered_results, wall_timeout_triggered


def _build_search_queries(user_query: str) -> List[str]:
    base = " ".join(user_query.strip().split())
    if not base:
        base = "industry trend analysis"

    candidates = [
        base,
        f"{base} latest statistics",
        f"{base} expert analysis",
        f"{base} case studies",
        f"{base} market outlook",
    ]
    queries = list(dict.fromkeys([item.strip() for item in candidates if item.strip()]))
    if len(queries) < _MIN_SEARCH_QUERIES:
        queries.extend(
            [
                "industry trend analysis latest statistics",
                "industry trend analysis expert analysis",
            ]
        )
        queries = list(dict.fromkeys(queries))
    return queries[:_MAX_SEARCH_QUERIES]


def _credibility_score(source: Mapping[str, Any]) -> float:
    url = source.get("url")
    provider = str(source.get("provider", "")).strip().lower()

    if isinstance(url, str) and url.strip():
        normalized = url.lower()
        if ".gov" in normalized or ".edu" in normalized:
            return 0.95
        if "wikipedia.org" in normalized:
            return 0.70
        return 0.80
    if provider == "perplexity":
        return 0.35
    return 0.45


def _fallback_snippet(query: str, provider: str, title: str) -> str:
    topic = query.strip() or "the requested topic"
    safe_title = title.strip() or "untitled source"
    return (
        f"{provider.upper()} fallback snippet for '{topic}' from '{safe_title}': "
        "detailed source excerpt was unavailable."
    )


def _normalize_source(
    raw: Mapping[str, Any], provider: str, query: str
) -> Dict[str, Any]:
    title = str(raw.get("title", "")).strip() or "Untitled source"
    raw_snippet = str(raw.get("snippet", "")).strip()
    snippet_from_provider = bool(raw_snippet)
    snippet = raw_snippet or _fallback_snippet(
        query=query, provider=provider, title=title
    )

    url_raw = raw.get("url")
    url = str(url_raw).strip() if isinstance(url_raw, str) and url_raw.strip() else None

    citation_available = bool(url)
    if provider == "perplexity":
        citation_available = bool(raw.get("citation_available", False)) and bool(url)

    source = {
        "title": title,
        "url": url,
        "snippet": snippet,
        "provider": provider,
        "citation_available": citation_available,
        "_snippet_from_provider": snippet_from_provider,
    }
    source["credibility_score"] = _credibility_score(source)
    return source


def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for source in sources:
        url = source.get("url")
        title = str(source.get("title", "")).strip().lower()
        key = f"url:{url.lower()}" if isinstance(url, str) and url else f"title:{title}"
        if key not in deduped:
            deduped[key] = source
            order.append(key)
            continue
        if float(source.get("credibility_score", 0.0)) > float(
            deduped[key].get("credibility_score", 0.0)
        ):
            deduped[key] = source
    return [deduped[key] for key in order]


def _has_meaningful_provider_snippet(sources: List[Dict[str, Any]]) -> bool:
    for source in sources:
        snippet_from_provider = source.get("_snippet_from_provider", None)
        is_provider_text = (
            True if snippet_from_provider is None else bool(snippet_from_provider)
        )
        if is_provider_text and len(str(source.get("snippet", "")).strip()) >= 20:
            return True
    return False


def _is_degraded(sources: List[Dict[str, Any]]) -> bool:
    if not sources:
        return True
    return not _has_meaningful_provider_snippet(sources)


def _parse_query_suggestions(
    response: Mapping[str, Any], fallback_query: str
) -> List[str]:
    raw = response.get("output", "")
    if not isinstance(raw, str) or not raw.strip():
        return _build_search_queries(fallback_query)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _build_search_queries(fallback_query)

    suggestions = []
    if isinstance(parsed, dict):
        suggestions = _safe_list(parsed.get("queries"))
    elif isinstance(parsed, list):
        suggestions = parsed

    cleaned = [str(item).strip() for item in suggestions if str(item).strip()]
    cleaned = list(dict.fromkeys(cleaned))
    if len(cleaned) < _MIN_SEARCH_QUERIES:
        return _build_search_queries(fallback_query)
    return cleaned[:_MAX_SEARCH_QUERIES]


def _synthesize_summary(
    query: str,
    sources: List[Dict[str, Any]],
    cost_controls: Mapping[str, Any],
) -> tuple[str, Dict[str, Any], bool, int, int]:
    if not sources:
        return (
            _deterministic_research_summary(query=query, sources=sources),
            {},
            True,
            0,
            0,
        )

    top = sources[:5]
    bullets = "\n".join(
        [f"- {item.get('title')}: {item.get('snippet')}" for item in top]
    )
    prompt = (
        "Synthesize a concise research brief from these findings.\n"
        f"Topic: {query}\n"
        f"Findings:\n{bullets}"
    )
    summary_started_at = perf_counter()
    llm_response = _safe_dict(
        generate_text(
            prompt=prompt,
            agent_key="research_agent",
            model=preferred_text_model(cost_controls, agent_key="research_agent"),
        )
    )
    summary_provider_latency_ms = max(
        0,
        int((perf_counter() - summary_started_at) * 1000),
    )
    summary = str(_safe_dict(llm_response).get("output", "")).strip()
    if summary:
        return summary, llm_response, False, summary_provider_latency_ms, 1
    return (
        _deterministic_research_summary(
            query=query,
            sources=sources,
        ),
        llm_response,
        True,
        summary_provider_latency_ms,
        1,
    )


def _make_degraded_perplexity_source(query: str) -> Dict[str, Any]:
    title = f"Perplexity fallback summary for: {query or 'requested topic'}"
    source = {
        "title": title,
        "url": None,
        "snippet": _fallback_snippet(query=query, provider="perplexity", title=title),
        "provider": "perplexity",
        "citation_available": False,
        "_snippet_from_provider": False,
    }
    source["credibility_score"] = _credibility_score(source)
    return source


def _tokenize_query(query: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def _is_low_value_keyword(token: str) -> bool:
    normalized = str(token).strip().lower()
    if not normalized:
        return True
    if normalized in _KEYWORD_PRESERVE_TOKENS:
        return False
    if len(normalized) < 2:
        return True
    return normalized in _KEYWORD_STOPWORDS


def _ensure_min_items(
    items: List[str], query: str, fallback_pool: List[str]
) -> List[str]:
    deduped = list(dict.fromkeys([item.strip() for item in items if item.strip()]))
    idx = 0
    query_seed = " ".join(_tokenize_query(query)[:4]) or "topic"
    while len(deduped) < _MIN_LIST_ITEMS:
        base = fallback_pool[idx % len(fallback_pool)]
        suffix = (
            f" ({query_seed})" if base not in deduped else f" ({query_seed}-{idx + 1})"
        )
        candidate = base if len(deduped) >= len(fallback_pool) else base + suffix
        if candidate not in deduped:
            deduped.append(candidate)
        idx += 1
    return deduped


def _fallback_summary(query: str, quality: str) -> str:
    topic = query.strip() or "the requested topic"
    if quality == "degraded":
        return (
            f"Degraded synthesis for '{topic}': source snippets were limited, "
            "so this summary uses deterministic fallback analysis."
        )
    return f"Synthesized research summary for '{topic}' based on collected sources."


def _extract_domain(url: str) -> str:
    raw = str(url).strip().lower()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    parsed = urlparse(raw)
    return str(parsed.hostname or "").strip().lower()


def _representative_domains(sources: List[Dict[str, Any]]) -> List[str]:
    domains: List[str] = []
    seen: set[str] = set()
    for source in sources:
        domain = _extract_domain(str(source.get("url", "")))
        if not domain or domain in seen:
            continue
        seen.add(domain)
        domains.append(domain)
        if len(domains) >= 4:
            break
    return domains


def _source_theme_candidates(sources: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for source in sources[:8]:
        title = str(source.get("title", "")).strip()
        snippet = str(source.get("snippet", "")).strip()
        if title:
            parts.append(title.lower())
        if snippet:
            parts.append(snippet.lower())
    return " ".join(parts)


def _detect_retrieved_themes(sources: List[Dict[str, Any]]) -> List[str]:
    haystack = _source_theme_candidates(sources)
    if not haystack:
        return []

    themes: List[str] = []
    for markers, label in _THEME_RULES:
        if any(marker in haystack for marker in markers):
            themes.append(label)

    if not themes:
        themes.append("directional trends synthesized from retrieved sources")
    return themes[:5]


def _truncate_single_line(value: str, limit: int = 180) -> str:
    text = " ".join(value.split()).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped}..."


def _deterministic_source_leads(sources: List[Dict[str, Any]]) -> List[str]:
    leads: List[str] = []
    for source in sources:
        title = str(source.get("title", "")).strip() or "Source"
        snippet = str(source.get("snippet", "")).strip()
        if not snippet:
            continue
        leads.append(f"{title} — {_truncate_single_line(snippet)}")
        if len(leads) >= 3:
            break
    return leads


def _deterministic_research_summary(
    *,
    query: str,
    sources: List[Dict[str, Any]],
) -> str:
    if not sources:
        topic = query.strip() or "the requested topic"
        return (
            "## Research Summary\n\n"
            "Text synthesis was unavailable, and no usable sources were retrieved for "
            f"'{topic}'. Try rerunning when providers are available."
        )

    citation_ready = sum(
        1 for source in sources if bool(source.get("citation_available", False))
    )
    domains = _representative_domains(sources)
    themes = _detect_retrieved_themes(sources)
    source_leads = _deterministic_source_leads(sources)

    lines = [
        "## Research Summary",
        "",
        (
            "Text synthesis was unavailable, so this is a deterministic summary "
            "from retrieved sources."
        ),
        "",
        "### Source Coverage",
        f"- Sources reviewed: {len(sources)}",
        f"- Citation-ready sources: {citation_ready}",
    ]
    if domains:
        lines.append(f"- Representative domains: {', '.join(domains)}")

    lines.extend(["", "### Retrieved Themes"])
    lines.extend([f"- {theme}" for theme in themes])

    if source_leads:
        lines.extend(["", "### Useful Source Leads"])
        lines.extend(
            [f"{index}. {lead}" for index, lead in enumerate(source_leads, start=1)]
        )

    return "\n".join(lines).strip()


def _build_key_facts(
    query: str, sources: List[Dict[str, Any]], quality: str
) -> List[str]:
    facts: List[str] = []
    for source in sources[:5]:
        title = str(source.get("title", "")).strip() or "Source"
        snippet = str(source.get("snippet", "")).strip()
        if not snippet:
            snippet = _fallback_snippet(
                query=query,
                provider=str(source.get("provider", "source")),
                title=title,
            )
        facts.append(f"{title}: {snippet}")
        if len(facts) >= _MIN_LIST_ITEMS:
            break

    if quality == "degraded":
        fallback_facts = [
            (
                f"Signals for '{query or 'the topic'}' are based on limited "
                "citation snippets."
            ),
            "Coverage focuses on directional trends rather than fully cited detail.",
            "Additional validation is recommended before publication decisions.",
        ]
    else:
        fallback_facts = [
            f"Multiple sources were aggregated for '{query or 'the topic'}'.",
            "Credibility scoring prioritized sources with stronger provenance.",
            "Findings were synthesized into a concise actionable brief.",
        ]
    return _ensure_min_items(facts, query, fallback_facts)


def _build_keywords(query: str, sources: List[Dict[str, Any]]) -> List[str]:
    words = [
        token
        for token in _tokenize_query(query)
        if not _is_low_value_keyword(token)
    ]
    from_titles: List[str] = []
    for source in sources[:5]:
        from_titles.extend(
            [
                token
                for token in _tokenize_query(str(source.get("title", "")))
                if not _is_low_value_keyword(token)
            ]
        )

    candidates = words + from_titles
    keywords: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        token = str(candidate).strip().lower()
        if _is_low_value_keyword(token) or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    if not keywords:
        keywords = [token.replace(" ", "_") for token in _FALLBACK_KEYWORDS]
    return _ensure_min_items(
        keywords, query, [token.replace(" ", "_") for token in _FALLBACK_KEYWORDS]
    )


def _build_entities(query: str, keywords: List[str]) -> List[str]:
    original_tokens = [token for token in query.split() if token and token[0].isupper()]
    entities = list(dict.fromkeys(original_tokens))
    if not entities:
        entities = [keyword.replace("_", " ").title() for keyword in keywords[:3]]
    return entities


def _sanitize_sources_for_output(
    query: str, sources: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for source in sources:
        entry = dict(source)
        title = str(entry.get("title", "")).strip() or "Untitled source"
        provider = str(entry.get("provider", "")).strip().lower() or "serp_api"

        if provider == "perplexity":
            entry["url"] = None
            entry["citation_available"] = False
        else:
            url = entry.get("url")
            entry["citation_available"] = bool(isinstance(url, str) and url.strip())

        snippet = str(entry.get("snippet", "")).strip()
        if not snippet:
            snippet = _fallback_snippet(query=query, provider=provider, title=title)
        entry["snippet"] = snippet
        entry["title"] = title
        entry["provider"] = provider
        entry.pop("_snippet_from_provider", None)
        sanitized.append(entry)
    return sanitized


def _degraded_research_payload(query: str, reason: str = "") -> Dict[str, Any]:
    sources: List[Dict[str, Any]] = []
    quality = "degraded"
    summary = _fallback_summary(query=query, quality=quality)
    keywords = _build_keywords(query=query, sources=sources)
    key_facts = _build_key_facts(query=query, sources=sources, quality=quality)
    entities = _build_entities(query=query, keywords=keywords)
    payload = {
        "status": "degraded",
        "degraded": True,
        "quality": quality,
        "cache_hit": False,
        "fallback_used": False,
        "queries": [],
        "query_count": 0,
        "source_count": 0,
        "synthesized_summary": summary,
        "summary": summary,
        "key_facts": key_facts,
        "keywords": keywords,
        "entities": entities,
    }
    if reason:
        payload["degraded_reason"] = reason
    return payload


def research_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run cache-first research retrieval with deterministic fallback behavior."""
    query = str(state.get("user_query", "")).strip()
    cache_key = build_research_cache_key(query, depth="standard")

    cached_payload = get_cached_research(state, cache_key)
    if isinstance(cached_payload, dict):
        cached_sources = _sanitize_sources_for_output(
            query=query,
            sources=deepcopy(_safe_list(cached_payload.get("sources"))),
        )
        cached_quality = (
            str(_safe_dict(cached_payload.get("research_data")).get("quality", ""))
            .strip()
            .lower()
        )
        if cached_quality not in {"standard", "degraded"}:
            cached_quality = "degraded" if _is_degraded(cached_sources) else "standard"
        cached_summary = str(
            _safe_dict(cached_payload.get("research_data")).get(
                "synthesized_summary", ""
            )
        ).strip() or _fallback_summary(query=query, quality=cached_quality)
        cached_keywords = _build_keywords(query=query, sources=cached_sources)
        cached_key_facts = _build_key_facts(
            query=query, sources=cached_sources, quality=cached_quality
        )
        cached_entities = _build_entities(query=query, keywords=cached_keywords)

        research_data = deepcopy(_safe_dict(cached_payload.get("research_data")))
        research_data["cache_hit"] = True
        research_data["quality"] = cached_quality
        research_data["synthesized_summary"] = cached_summary
        research_data["summary"] = cached_summary
        research_data["key_facts"] = cached_key_facts
        research_data["keywords"] = cached_keywords
        research_data["entities"] = cached_entities
        research_data["degraded"] = cached_quality == "degraded"
        research_data["status"] = (
            "degraded" if cached_quality == "degraded" else "complete"
        )
        updates = {
            "research_data": research_data,
            "sources": cached_sources,
            "workflow_status": "research_complete",
            "final_response": None,
        }
        updates.update(touch_cached_research_key(state, cache_key))
        return updates

    cost_controls = normalize_cost_controls(_safe_dict(state.get("cost_controls")))
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        return {
            "research_data": _degraded_research_payload(
                query, reason="token_budget_exceeded"
            ),
            "sources": [],
            "cost_controls": cost_controls,
            "workflow_status": "research_complete",
            "final_response": None,
        }
    if search_cap_reached(cost_controls):
        degraded_payload = _degraded_research_payload(
            query, reason="search_cap_reached"
        )
        degraded_payload["search_cap_reached"] = True
        return {
            "research_data": degraded_payload,
            "sources": [],
            "cost_controls": cost_controls,
            "workflow_status": "research_complete",
            "final_response": None,
        }

    used_queries = int(cost_controls.get("search_queries_used_this_session", 0))
    query_cap = int(
        cost_controls.get("search_query_cap_per_session", _DEFAULT_SEARCH_QUERY_CAP)
    )
    remaining_calls = max(0, query_cap - used_queries)
    provider_latency_total_ms = 0
    provider_call_count = 0
    provider_latency_by_provider_ms: Dict[str, int] = {}
    provider_call_count_by_provider: Dict[str, int] = {}
    provider_timeout_count = 0
    provider_timeout_count_by_provider: Dict[str, int] = {}
    provider_latency_wall_ms = 0
    search_provider_wall_timeout_triggered = False

    query_generation_prompt = (
        "Generate 3-5 search queries as JSON list for this topic:\n" f"{query}"
    )
    query_generation_model = preferred_text_model(
        cost_controls,
        agent_key="research_agent",
    )
    query_generation_started_at = perf_counter()
    query_generation = _safe_dict(
        generate_text(
            prompt=query_generation_prompt,
            agent_key="research_agent",
            model=query_generation_model,
        )
    )
    query_generation_provider_latency_ms = max(
        0,
        int((perf_counter() - query_generation_started_at) * 1000),
    )
    provider_latency_total_ms += query_generation_provider_latency_ms
    provider_call_count += 1
    provider_latency_wall_ms += query_generation_provider_latency_ms
    query_generation_provider = _provider_from_llm_response(
        query_generation,
        fallback_model=query_generation_model,
    )
    _increment_int_map(
        provider_latency_by_provider_ms,
        key=query_generation_provider,
        delta=query_generation_provider_latency_ms,
    )
    _increment_int_map(
        provider_call_count_by_provider,
        key=query_generation_provider,
        delta=1,
    )
    cost_controls = apply_text_tokens(cost_controls, query_generation)
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        degraded_payload = _degraded_research_payload(
            query,
            reason="token_budget_exceeded_after_query_planning",
        )
        degraded_payload["provider_latency_total_ms"] = (
            _safe_non_negative_int(provider_latency_total_ms) or 0
        )
        degraded_payload["provider_latency_wall_ms"] = (
            _safe_non_negative_int(provider_latency_wall_ms) or 0
        )
        degraded_payload["provider_latency_by_provider_ms"] = dict(
            provider_latency_by_provider_ms
        )
        degraded_payload["provider_call_count"] = provider_call_count
        degraded_payload["provider_call_count_by_provider"] = dict(
            provider_call_count_by_provider
        )
        degraded_payload["provider_timeout_count"] = provider_timeout_count
        degraded_payload["provider_timeout_count_by_provider"] = dict(
            provider_timeout_count_by_provider
        )
        degraded_payload["search_provider_wall_timeout_ms"] = int(
            _SEARCH_PROVIDER_WALL_TIMEOUT_SECONDS * 1000
        )
        degraded_payload["search_provider_wall_timeout_triggered"] = False
        return {
            "research_data": degraded_payload,
            "sources": [],
            "cost_controls": cost_controls,
            "workflow_status": "research_complete",
            "final_response": None,
        }

    search_queries = _parse_query_suggestions(query_generation, fallback_query=query)

    executed_queries: List[str] = []
    collected_sources: List[Dict[str, Any]] = []
    fallback_used = False
    search_calls_used = 0
    search_phase_started_at = perf_counter()
    search_provider_deadline = perf_counter() + _SEARCH_PROVIDER_WALL_TIMEOUT_SECONDS

    primary_queries = search_queries[:remaining_calls]
    primary_results, primary_wall_timeout_triggered = _run_search_fanout(
        queries=primary_queries,
        depth="standard",
        max_concurrency=_SEARCH_FANOUT_CONCURRENCY,
        per_query_timeout_seconds=_SEARCH_QUERY_TIMEOUT_SECONDS,
        node_deadline=search_provider_deadline,
    )
    search_provider_wall_timeout_triggered = (
        search_provider_wall_timeout_triggered or primary_wall_timeout_triggered
    )

    degraded_primary_queries: List[str] = []
    for primary_result in primary_results:
        search_query = primary_result.query
        if primary_result.timed_out:
            provider_timeout_count += 1
            _increment_int_map(provider_timeout_count_by_provider, key="serp_api")
        if not primary_result.attempted:
            fallback_used = True
            degraded_primary_queries.append(search_query)
            continue

        provider_latency_total_ms += primary_result.duration_ms
        provider_call_count += 1
        _increment_int_map(
            provider_latency_by_provider_ms,
            key="serp_api",
            delta=primary_result.duration_ms,
        )
        _increment_int_map(provider_call_count_by_provider, key="serp_api", delta=1)
        remaining_calls = max(0, remaining_calls - 1)
        search_calls_used += 1
        executed_queries.append(search_query)

        primary_raw_results = _safe_list(primary_result.response.get("results"))
        primary_sources = [
            _normalize_source(item, provider="serp_api", query=search_query)
            for item in primary_raw_results
            if isinstance(item, Mapping)
        ]

        if not _is_degraded(primary_sources):
            collected_sources.extend(primary_sources)
            continue

        fallback_used = True
        degraded_primary_queries.append(search_query)

    fallback_budget = max(0, remaining_calls)
    fallback_queries = degraded_primary_queries[:fallback_budget]
    fallback_skipped_queries = degraded_primary_queries[fallback_budget:]

    fallback_results, fallback_wall_timeout_triggered = _run_search_fanout(
        queries=fallback_queries,
        depth="fallback",
        max_concurrency=_SEARCH_FANOUT_CONCURRENCY,
        per_query_timeout_seconds=_SEARCH_QUERY_TIMEOUT_SECONDS,
        node_deadline=search_provider_deadline,
    )
    search_provider_wall_timeout_triggered = (
        search_provider_wall_timeout_triggered or fallback_wall_timeout_triggered
    )

    for fallback_result in fallback_results:
        search_query = fallback_result.query
        if fallback_result.timed_out:
            provider_timeout_count += 1
            _increment_int_map(provider_timeout_count_by_provider, key="perplexity")
        if not fallback_result.attempted:
            collected_sources.append(_make_degraded_perplexity_source(search_query))
            continue

        provider_latency_total_ms += fallback_result.duration_ms
        provider_call_count += 1
        _increment_int_map(
            provider_latency_by_provider_ms,
            key="perplexity",
            delta=fallback_result.duration_ms,
        )
        _increment_int_map(provider_call_count_by_provider, key="perplexity", delta=1)
        remaining_calls = max(0, remaining_calls - 1)
        search_calls_used += 1

        fallback_raw_results = _safe_list(fallback_result.response.get("results"))
        fallback_sources = [
            _normalize_source(item, provider="perplexity", query=search_query)
            for item in fallback_raw_results
            if isinstance(item, Mapping)
        ]
        if not fallback_sources:
            fallback_sources = [_make_degraded_perplexity_source(search_query)]
        collected_sources.extend(fallback_sources)

    for search_query in fallback_skipped_queries:
        collected_sources.append(_make_degraded_perplexity_source(search_query))

    search_provider_wall_elapsed_ms = max(
        0,
        int((perf_counter() - search_phase_started_at) * 1000),
    )
    provider_latency_wall_ms += search_provider_wall_elapsed_ms

    deduped_sources = _dedupe_sources(collected_sources)
    deduped_sources.sort(
        key=lambda item: float(item.get("credibility_score", 0.0)), reverse=True
    )
    degraded = _is_degraded(deduped_sources) or len(executed_queries) == 0

    deduped_sources = _sanitize_sources_for_output(query=query, sources=deduped_sources)
    quality = "degraded" if degraded else "standard"
    (
        summary,
        summary_response,
        deterministic_summary_used,
        summary_provider_latency_ms,
        summary_provider_call_count,
    ) = _synthesize_summary(
        query=query,
        sources=deduped_sources,
        cost_controls=cost_controls,
    )
    provider_latency_total_ms += summary_provider_latency_ms
    provider_call_count += summary_provider_call_count
    provider_latency_wall_ms += summary_provider_latency_ms
    if summary_provider_call_count > 0:
        summary_provider = _provider_from_llm_response(
            summary_response,
            fallback_model=preferred_text_model(
                cost_controls,
                agent_key="research_agent",
            ),
        )
        _increment_int_map(
            provider_latency_by_provider_ms,
            key=summary_provider,
            delta=summary_provider_latency_ms,
        )
        _increment_int_map(
            provider_call_count_by_provider,
            key=summary_provider,
            delta=summary_provider_call_count,
        )
    cost_controls = apply_text_tokens(cost_controls, summary_response)
    if not summary.strip():
        summary = _deterministic_research_summary(query=query, sources=deduped_sources)
        deterministic_summary_used = True

    key_facts = _build_key_facts(query=query, sources=deduped_sources, quality=quality)
    keywords = _build_keywords(query=query, sources=deduped_sources)
    entities = _build_entities(query=query, keywords=keywords)

    research_data = {
        "status": "degraded" if degraded else "complete",
        "degraded": degraded,
        "quality": quality,
        "cache_hit": False,
        "fallback_used": fallback_used,
        "queries": executed_queries,
        "query_count": len(executed_queries),
        "source_count": len(deduped_sources),
        "synthesized_summary": summary,
        "summary": summary,
        "key_facts": key_facts,
        "keywords": keywords,
        "entities": entities,
        "deterministic_summary_used": deterministic_summary_used,
    }
    if provider_call_count > 0:
        research_data["provider_latency_total_ms"] = (
            _safe_non_negative_int(provider_latency_total_ms) or 0
        )
        research_data["provider_latency_wall_ms"] = (
            _safe_non_negative_int(provider_latency_wall_ms) or 0
        )
        research_data["provider_latency_by_provider_ms"] = dict(
            provider_latency_by_provider_ms
        )
        research_data["provider_call_count"] = provider_call_count
        research_data["provider_call_count_by_provider"] = dict(
            provider_call_count_by_provider
        )
    research_data["provider_timeout_count"] = provider_timeout_count
    research_data["provider_timeout_count_by_provider"] = dict(
        provider_timeout_count_by_provider
    )
    research_data["search_provider_wall_timeout_ms"] = int(
        _SEARCH_PROVIDER_WALL_TIMEOUT_SECONDS * 1000
    )
    research_data["search_provider_wall_timeout_triggered"] = bool(
        search_provider_wall_timeout_triggered
    )

    cost_controls["search_queries_used_this_session"] = used_queries + search_calls_used
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True

    updates: Dict[str, Any] = {
        "research_data": research_data,
        "sources": deduped_sources,
        "cost_controls": cost_controls,
        "workflow_status": "research_complete",
        "final_response": None,
    }

    if not degraded and deduped_sources:
        cacheable_research_data = deepcopy(research_data)
        cacheable_research_data.pop("queries", None)
        cache_payload = {
            "research_data": cacheable_research_data,
            "sources": deepcopy(deduped_sources),
        }
        updates.update(set_cached_research(state, cache_key, cache_payload))

    return updates
