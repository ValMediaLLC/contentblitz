"""Blog writer node implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping

from contentblitz.core.cost_controls import (
    apply_text_tokens,
    normalize_cost_controls,
    preferred_text_model,
    token_budget_exceeded,
)
from contentblitz.tools.text import generate_text


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _build_prompt(
    user_query: str,
    blog_brief: Mapping[str, Any],
    citations: List[Mapping[str, Any]],
    retry_feedback: List[str],
) -> str:
    objective = (
        str(blog_brief.get("objective", "")).strip() or "Create an SEO-focused article."
    )
    audience = str(blog_brief.get("audience", "")).strip() or "professional audience"
    tone = str(blog_brief.get("tone", "")).strip() or "clear and practical"
    angle = str(blog_brief.get("angle", "")).strip() or "educational narrative"
    outline = blog_brief.get("outline", [])
    outline_lines = []
    if isinstance(outline, list):
        outline_lines = [
            f"- {str(item).strip()}" for item in outline if str(item).strip()
        ]
    citation_lines = [
        f"- {str(item.get('title', 'Source')).strip()} ({item.get('url')})"
        for item in citations
    ]
    feedback_lines = [f"- {item}" for item in retry_feedback if str(item).strip()]

    prompt = (
        "Write an SEO-friendly blog draft in markdown.\n"
        f"Topic: {user_query or 'Requested topic'}\n"
        f"Objective: {objective}\n"
        f"Audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Angle: {angle}\n"
    )
    if outline_lines:
        prompt += "Outline:\n" + "\n".join(outline_lines) + "\n"
    if citation_lines:
        prompt += (
            "Approved citations (use only these):\n" + "\n".join(citation_lines) + "\n"
        )
    if feedback_lines:
        prompt += "Retry feedback to address:\n" + "\n".join(feedback_lines) + "\n"
    prompt += "Return only the draft body text."
    return prompt


def _fallback_draft(user_query: str, blog_brief: Mapping[str, Any]) -> str:
    objective = str(
        blog_brief.get("objective", "Create an informative article.")
    ).strip()
    angle = str(blog_brief.get("angle", "practical guidance")).strip()
    title = user_query.strip() or "Strategic Content Planning"
    return (
        f"# {title}\n\n"
        f"{objective}\n\n"
        f"This draft takes a {angle} approach, focusing on clear steps readers can apply immediately."
    )


def _render_citations(citations: List[Mapping[str, Any]]) -> str:
    if not citations:
        return ""
    lines = ["## Sources"]
    for index, source in enumerate(citations, start=1):
        title = str(source.get("title", "Source")).strip() or "Source"
        url = str(source.get("url", "")).strip()
        lines.append(f"[{index}] {title} ({url})")
    return "\n".join(lines)


def _word_count(text: str) -> int:
    return len([token for token in text.split() if token.strip()])


def _readability_score(text: str) -> float:
    words = _word_count(text)
    if words == 0:
        return 0.0
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
    avg_sentence_len = words / sentence_count
    # Deterministic proxy score (higher easier), capped [0, 100].
    score = max(0.0, min(100.0, 100.0 - (avg_sentence_len * 1.3)))
    return round(score, 2)


def _append_budget_error(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors = deepcopy(_safe_list(state.get("errors", [])))
    errors.append(
        {
            "agent": "blog_writer",
            "type": "budget_exceeded",
            "message": "Blog generation used deterministic fallback due to token budget limits.",
            "recoverable": True,
        }
    )
    return errors


def blog_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a blog draft using content brief + available citations."""
    content_brief = _safe_dict(state.get("content_brief", {}))
    blog_brief = _safe_dict(content_brief.get("blog", {}))
    user_query = str(state.get("user_query", "")).strip()

    content_drafts = _safe_dict(state.get("content_drafts", {}))
    existing_blog = _safe_dict(content_drafts.get("blog", {}))
    current_version = int(existing_blog.get("version", 0))
    next_version = current_version + 1

    sources = [
        item
        for item in _safe_list(state.get("sources", []))
        if isinstance(item, Mapping)
    ]
    citations = [
        source
        for source in sources
        if bool(source.get("citation_available", False))
        and isinstance(source.get("url"), str)
        and str(source.get("url", "")).strip()
    ]

    retry_feedback = []
    retry_state = _safe_dict(state.get("retry_feedback", {}))
    blog_feedback = retry_state.get("blog", [])
    if isinstance(blog_feedback, list):
        retry_feedback = [
            str(item).strip() for item in blog_feedback if str(item).strip()
        ]

    cost_controls = normalize_cost_controls(_safe_dict(state.get("cost_controls", {})))
    draft_status = _safe_dict(state.get("draft_status", {}))

    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        draft_body = _fallback_draft(user_query=user_query, blog_brief=blog_brief)
        citations_block = _render_citations(citations)
        if citations_block:
            draft_body = f"{draft_body}\n\n{citations_block}"
        else:
            draft_body = (
                f"{draft_body}\n\n"
                "Disclaimer: No verifiable external citations were available for this draft."
            )

        blog_update = {
            **existing_blog,
            "body": draft_body,
            "version": next_version,
            "word_count": _word_count(draft_body),
            "readability_score": _readability_score(draft_body),
            "model_used": "budget_fallback",
        }
        return {
            "content_drafts": {"blog": blog_update},
            "draft_status": {
                **draft_status,
                "blog": "complete",
            },
            "cost_controls": cost_controls,
            "errors": _append_budget_error(state),
        }

    model = preferred_text_model(cost_controls)
    prompt = _build_prompt(
        user_query=user_query,
        blog_brief=blog_brief,
        citations=citations,
        retry_feedback=retry_feedback,
    )

    llm_response = _safe_dict(
        generate_text(
            prompt=prompt,
            agent_key="blog_writer",
            model=model,
        )
    )
    draft_body = str(llm_response.get("output", "")).strip()
    if not draft_body:
        draft_body = _fallback_draft(user_query=user_query, blog_brief=blog_brief)

    citations_block = _render_citations(citations)
    if citations_block:
        draft_body = f"{draft_body}\n\n{citations_block}"
    else:
        draft_body = (
            f"{draft_body}\n\n"
            "Disclaimer: No verifiable external citations were available for this draft."
        )

    cost_controls = apply_text_tokens(cost_controls, llm_response)
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True

    blog_update = {
        **existing_blog,
        "body": draft_body,
        "version": next_version,
        "word_count": _word_count(draft_body),
        "readability_score": _readability_score(draft_body),
        "model_used": model,
    }

    return {
        "content_drafts": {"blog": blog_update},
        "draft_status": {
            **draft_status,
            "blog": "complete",
        },
        "cost_controls": cost_controls,
    }
