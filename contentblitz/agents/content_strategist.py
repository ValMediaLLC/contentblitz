"""Content strategist node implementation."""

from __future__ import annotations

import json
from copy import deepcopy
from time import perf_counter
from typing import Any, Dict, List, Mapping

from contentblitz.core.cost_controls import (
    apply_text_tokens,
    normalize_cost_controls,
    preferred_text_model,
    token_budget_exceeded,
)
from contentblitz.tools.text import generate_text

_SUPPORTED_OUTPUTS = {"blog", "linkedin", "image", "research"}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize_outputs(outputs: Any) -> List[str]:
    raw = _safe_list(outputs)
    normalized = [str(item).strip().lower() for item in raw if str(item).strip()]
    deduped = list(dict.fromkeys(normalized))
    return [item for item in deduped if item in _SUPPORTED_OUTPUTS]


def _fallback_brief(
    output_type: str,
    user_query: str,
    intent: str,
    brand_voice: Mapping[str, Any],
    research_data: Mapping[str, Any],
) -> Dict[str, Any]:
    tone = (
        str(brand_voice.get("tone", "clear and practical")).strip()
        or "clear and practical"
    )
    audience = (
        str(brand_voice.get("audience", "professional audience")).strip()
        or "professional audience"
    )
    summary = (
        str(research_data.get("synthesized_summary", "")).strip()
        or str(research_data.get("summary", "")).strip()
    )
    if not summary:
        summary = f"Research context for '{user_query or 'the topic'}' is limited."

    safe_intent = str(intent).strip().replace("_", " ")
    requested_topic = user_query or "the requested topic"
    if safe_intent:
        objective = (
            f"Requested deliverable: {requested_topic}. "
            f"Intent focus: {safe_intent}."
        )
    else:
        objective = f"Requested deliverable: {requested_topic}."

    common = {
        "objective": objective,
        "audience": audience,
        "tone": tone,
        "research_anchor": summary,
    }

    if output_type == "blog":
        return {
            **common,
            "format": "blog",
            "angle": "educational narrative with practical takeaways",
            "outline": [
                "Hook with the core problem",
                "Explain current trend and implications",
                "Conclude with actionable guidance",
            ],
        }
    if output_type == "linkedin":
        return {
            **common,
            "format": "linkedin",
            "angle": "opinionated insight with concise credibility points",
            "structure": [
                "Opening insight",
                "Two supporting points",
                "Invitation for discussion",
            ],
        }
    return {
        **common,
        "format": "image",
        "angle": "single-scene visual concept",
        "visual_direction": "clean, high contrast, modern composition",
        "prompt_focus": (
            f"Visualize '{user_query or 'the topic'}' with strategic clarity."
        ),
    }


def _parse_brief_output(
    llm_output: Any,
    output_type: str,
    user_query: str,
    intent: str,
    brand_voice: Mapping[str, Any],
    research_data: Mapping[str, Any],
) -> Dict[str, Any]:
    payload: Any = None
    if isinstance(llm_output, dict):
        payload = llm_output
    elif isinstance(llm_output, str) and llm_output.strip():
        try:
            payload = json.loads(llm_output)
        except json.JSONDecodeError:
            payload = None

    if isinstance(payload, Mapping):
        brief = dict(payload)
        brief.setdefault("format", output_type)
        return brief

    return _fallback_brief(
        output_type=output_type,
        user_query=user_query,
        intent=intent,
        brand_voice=brand_voice,
        research_data=research_data,
    )


def _build_brief_prompt(
    output_type: str,
    user_query: str,
    intent: str,
    brand_voice: Mapping[str, Any],
    research_data: Mapping[str, Any],
    sources: List[Mapping[str, Any]],
) -> str:
    summary = (
        str(research_data.get("synthesized_summary", "")).strip()
        or str(research_data.get("summary", "")).strip()
    )
    tone = (
        str(brand_voice.get("tone", "clear and practical")).strip()
        or "clear and practical"
    )
    audience = (
        str(brand_voice.get("audience", "professional audience")).strip()
        or "professional audience"
    )
    return (
        f"Create a JSON content brief for '{output_type}'.\n"
        f"User query: {user_query}\n"
        f"Intent: {intent}\n"
        f"Brand voice tone: {tone}\n"
        f"Audience: {audience}\n"
        f"Research summary: {summary}\n"
        f"Source count: {len(sources)}\n"
        "Return JSON only."
    )


def _build_research_report(
    user_query: str,
    research_data: Mapping[str, Any],
    sources: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = (
        str(research_data.get("synthesized_summary", "")).strip()
        or str(research_data.get("summary", "")).strip()
    )
    if not summary:
        summary = f"Research context for '{user_query or 'the topic'}' is limited."
    title = f"Research Report: {user_query or 'Requested Topic'}"
    if summary.lstrip().startswith("## Research Summary"):
        body = summary
    else:
        body = (
            f"Research report for '{user_query or 'the requested topic'}'. "
            f"{summary} Sources reviewed: {len(sources)}."
        )
    sections = [
        "Executive Summary",
        "Key Findings",
        "Source Notes",
    ]
    return {
        "title": title,
        "body": body,
        "sections": sections,
    }


def content_strategist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate channel briefs from research context without writing final drafts."""
    outputs = _normalize_outputs(state.get("requested_outputs", []))
    user_query = str(state.get("user_query", "")).strip()
    intent = str(state.get("intent", "")).strip()
    brand_voice = _safe_dict(state.get("brand_voice", {}))
    research_data = _safe_dict(state.get("research_data", {}))
    sources = [
        item
        for item in _safe_list(state.get("sources", []))
        if isinstance(item, Mapping)
    ]

    content_brief = deepcopy(_safe_dict(state.get("content_brief", {})))
    content_brief.setdefault("blog", {})
    content_brief.setdefault("linkedin", {})
    content_brief.setdefault("image", {})

    cost_controls = normalize_cost_controls(_safe_dict(state.get("cost_controls", {})))
    provider_latency_total_ms = 0
    provider_call_count = 0
    provider_name = ""
    model_used = ""

    for output_type in ("blog", "linkedin", "image"):
        if output_type not in outputs:
            continue
        if token_budget_exceeded(cost_controls):
            cost_controls["budget_exceeded"] = True
            content_brief[output_type] = _fallback_brief(
                output_type=output_type,
                user_query=user_query,
                intent=intent,
                brand_voice=brand_voice,
                research_data=research_data,
            )
            continue
        prompt = _build_brief_prompt(
            output_type=output_type,
            user_query=user_query,
            intent=intent,
            brand_voice=brand_voice,
            research_data=research_data,
            sources=sources,
        )
        provider_started_at = perf_counter()
        llm_response = _safe_dict(
            generate_text(
                prompt=prompt,
                agent_key="content_strategist",
                model=preferred_text_model(cost_controls),
            )
        )
        provider_latency_total_ms += max(
            0,
            int((perf_counter() - provider_started_at) * 1000),
        )
        provider_call_count += 1
        provider_candidate = str(llm_response.get("provider", "")).strip().lower()
        model_candidate = str(llm_response.get("model", "")).strip()
        if provider_candidate:
            provider_name = provider_candidate
        if model_candidate:
            model_used = model_candidate
        cost_controls = apply_text_tokens(cost_controls, llm_response)
        content_brief[output_type] = _parse_brief_output(
            llm_output=llm_response.get("output", ""),
            output_type=output_type,
            user_query=user_query,
            intent=intent,
            brand_voice=brand_voice,
            research_data=research_data,
        )

    updates: Dict[str, Any] = {
        "content_brief": content_brief,
        "cost_controls": cost_controls,
        "workflow_status": "strategy_complete",
        "final_response": None,
    }
    if provider_call_count > 0:
        tool_outputs = deepcopy(_safe_dict(state.get("tool_outputs", {})))
        strategist_metrics: Dict[str, Any] = {
            "provider_call_count": provider_call_count,
            "provider_latency_ms": max(0, int(provider_latency_total_ms)),
        }
        if provider_name:
            strategist_metrics["provider"] = provider_name
        if model_used:
            strategist_metrics["model"] = model_used
        tool_outputs["content_strategist"] = strategist_metrics
        updates["tool_outputs"] = tool_outputs

    if "research" in outputs and any(
        item in outputs for item in ("blog", "linkedin", "image")
    ):
        content_drafts = deepcopy(_safe_dict(state.get("content_drafts", {})))
        content_drafts.setdefault("research_report", {"body": ""})
        content_drafts["research_report"] = _build_research_report(
            user_query=user_query,
            research_data=research_data,
            sources=sources,
        )
        updates["content_drafts"] = content_drafts

    return updates
