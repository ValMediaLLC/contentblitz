"""Research agent node implementation for Phase 1/2 placeholder flow."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping

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
_MIN_LIST_ITEMS = 3
_FALLBACK_KEYWORDS = ["market trends", "audience insights", "strategic positioning"]


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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
) -> tuple[str, Dict[str, Any]]:
    if not sources:
        return f"Limited research results were found for '{query}'.", {}

    top = sources[:5]
    bullets = "\n".join(
        [f"- {item.get('title')}: {item.get('snippet')}" for item in top]
    )
    prompt = (
        "Synthesize a concise research brief from these findings.\n"
        f"Topic: {query}\n"
        f"Findings:\n{bullets}"
    )
    llm_response = _safe_dict(
        generate_text(
            prompt=prompt,
            agent_key="research_agent",
            model=preferred_text_model(cost_controls),
        )
    )
    summary = str(_safe_dict(llm_response).get("output", "")).strip()
    if summary:
        return summary, llm_response
    return (
        f"Research findings compiled for '{query}' from {len(sources)} sources.",
        llm_response,
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


# TODO(architecture):
# Improve deterministic keyword extraction by filtering common stopwords
# and normalizing singular/plural entities before passing research_data
# into downstream content strategy agents.
def _build_keywords(query: str, sources: List[Dict[str, Any]]) -> List[str]:
    words = [token for token in _tokenize_query(query) if len(token) >= 3]
    from_titles: List[str] = []
    for source in sources[:5]:
        from_titles.extend(
            [
                token
                for token in _tokenize_query(str(source.get("title", "")))
                if len(token) >= 4
            ]
        )

    candidates = words + from_titles
    keywords = list(dict.fromkeys(candidates))
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

    query_generation_prompt = (
        "Generate 3-5 search queries as JSON list for this topic:\n" f"{query}"
    )
    query_generation = generate_text(
        prompt=query_generation_prompt,
        agent_key="research_agent",
        model=preferred_text_model(cost_controls),
    )
    cost_controls = apply_text_tokens(cost_controls, _safe_dict(query_generation))
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        return {
            "research_data": _degraded_research_payload(
                query,
                reason="token_budget_exceeded_after_query_planning",
            ),
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

    for search_query in search_queries:
        if remaining_calls <= 0:
            break

        try:
            primary_response = search_web(query=search_query, depth="standard")
        except Exception:
            primary_response = {"results": []}
        remaining_calls -= 1
        search_calls_used += 1
        executed_queries.append(search_query)

        primary_results = _safe_list(_safe_dict(primary_response).get("results"))
        primary_sources = [
            _normalize_source(item, provider="serp_api", query=search_query)
            for item in primary_results
            if isinstance(item, Mapping)
        ]

        if not _is_degraded(primary_sources):
            collected_sources.extend(primary_sources)
            continue

        fallback_used = True
        if remaining_calls <= 0:
            collected_sources.append(_make_degraded_perplexity_source(search_query))
            continue

        try:
            fallback_response = search_web(query=search_query, depth="fallback")
        except Exception:
            fallback_response = {"results": []}
        remaining_calls -= 1
        search_calls_used += 1

        fallback_results = _safe_list(_safe_dict(fallback_response).get("results"))
        fallback_sources = [
            _normalize_source(item, provider="perplexity", query=search_query)
            for item in fallback_results
            if isinstance(item, Mapping)
        ]
        if not fallback_sources:
            fallback_sources = [_make_degraded_perplexity_source(search_query)]
        collected_sources.extend(fallback_sources)

    deduped_sources = _dedupe_sources(collected_sources)
    deduped_sources.sort(
        key=lambda item: float(item.get("credibility_score", 0.0)), reverse=True
    )
    degraded = _is_degraded(deduped_sources) or len(executed_queries) == 0

    deduped_sources = _sanitize_sources_for_output(query=query, sources=deduped_sources)
    quality = "degraded" if degraded else "standard"
    summary, summary_response = _synthesize_summary(
        query=query,
        sources=deduped_sources,
        cost_controls=cost_controls,
    )
    cost_controls = apply_text_tokens(cost_controls, summary_response)
    if not summary.strip():
        summary = _fallback_summary(query=query, quality=quality)
    if quality == "degraded":
        summary = _fallback_summary(query=query, quality=quality)

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
    }

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
