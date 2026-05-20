"""LinkedIn writer node implementation."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping

from contentblitz.core.cost_controls import (
    apply_text_tokens,
    normalize_cost_controls,
    preferred_text_model,
    token_budget_exceeded,
)
from contentblitz.tools.text import generate_text

_MIN_LINKEDIN_CHARS = 1300
_MAX_LINKEDIN_CHARS = 1600
_FALLBACK_PROVIDER_WARNING = (
    "Draft unavailable because text generation is currently limited. "
    "Research sources were collected successfully and can be used to regenerate "
    "this section once the provider is available."
)

_CTA_HINTS = (
    "comment",
    "share",
    "reply",
    "follow",
    "dm",
    "tell me",
    "let me know",
    "what do you think",
    "drop",
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _build_prompt(
    user_query: str,
    linkedin_brief: Mapping[str, Any],
    retry_feedback: List[str],
    previous_char_count: int = 0,
) -> str:
    objective = (
        str(linkedin_brief.get("objective", "")).strip()
        or "Create a strong LinkedIn post."
    )
    audience = (
        str(linkedin_brief.get("audience", "")).strip() or "professional audience"
    )
    tone = str(linkedin_brief.get("tone", "")).strip() or "clear, direct, and practical"
    angle = (
        str(linkedin_brief.get("angle", "")).strip() or "insight-driven and actionable"
    )
    structure = linkedin_brief.get("structure", [])
    structure_lines = []
    if isinstance(structure, list):
        structure_lines = [
            f"- {str(item).strip()}" for item in structure if str(item).strip()
        ]
    feedback_lines = [f"- {item}" for item in retry_feedback if str(item).strip()]

    prompt = (
        "Write a LinkedIn post in plain text.\n"
        f"Topic: {user_query or 'Requested topic'}\n"
        f"Objective: {objective}\n"
        f"Audience: {audience}\n"
        f"Tone: {tone}\n"
        f"Angle: {angle}\n"
        "Constraints:\n"
        "- Include a strong opening hook.\n"
        "- Include one clear CTA near the end.\n"
        "- Include 3-6 relevant hashtags.\n"
        f"- Keep total length between {_MIN_LINKEDIN_CHARS}-{_MAX_LINKEDIN_CHARS} "
        "characters when possible.\n"
    )
    if previous_char_count > 0:
        prompt += (
            f"Previous draft length was {previous_char_count} characters. "
            "Expand with concrete detail and examples while staying concise.\n"
        )
    if structure_lines:
        prompt += "Suggested structure:\n" + "\n".join(structure_lines) + "\n"
    if feedback_lines:
        prompt += "Retry feedback to address:\n" + "\n".join(feedback_lines) + "\n"
    prompt += "Return only the post text."
    return prompt


def _fallback_hashtags(user_query: str) -> List[str]:
    words = re.findall(r"[A-Za-z0-9]+", user_query or "")
    tags: List[str] = []
    for word in words:
        cleaned = word.strip()
        if len(cleaned) < 3:
            continue
        tag = f"#{cleaned[0].upper()}{cleaned[1:]}"
        if tag.lower() not in {item.lower() for item in tags}:
            tags.append(tag)
        if len(tags) >= 6:
            break
    if len(tags) < 3:
        defaults = ["#MarketingStrategy", "#ContentOps", "#AIWorkflows"]
        for tag in defaults:
            if tag.lower() not in {item.lower() for item in tags}:
                tags.append(tag)
            if len(tags) >= 3:
                break
    return tags


def _extract_hashtags(text: str, user_query: str) -> List[str]:
    found = re.findall(r"#([A-Za-z0-9_]+)", text or "")
    tags: List[str] = []
    seen = set()
    for raw in found:
        tag = f"#{raw}"
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    if tags:
        return tags
    return _fallback_hashtags(user_query)


def _extract_hook(text: str, user_query: str) -> str:
    for line in (text or "").splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("#"):
            continue
        return clean

    _ = user_query
    return (
        "AI-driven content operations are changing faster than most teams can "
        "execute."
    )


def _extract_cta(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        lower = line.lower()
        if any(hint in lower for hint in _CTA_HINTS) or line.endswith("?"):
            return line
    return "What are you seeing in your own team right now? Share in the comments."


def _truncate_clean(text: str, limit: int) -> str:
    """Truncate at sentence or whitespace boundary, avoiding mid-word cuts."""
    if limit <= 0:
        return ""
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    candidate = text[:limit]

    sentence_matches = list(re.finditer(r"[.!?](?=\s|$)", candidate))
    if sentence_matches:
        return candidate[: sentence_matches[-1].end()].rstrip()

    whitespace_matches = list(re.finditer(r"\s+", candidate))
    if whitespace_matches:
        return candidate[: whitespace_matches[-1].start()].rstrip()

    return candidate.rstrip()


def _truncate_with_tail(text: str, tail: str, max_chars: int) -> str:
    """
    Truncate text cleanly while preserving CTA/hashtags tail whenever possible.
    """
    if len(text) <= max_chars:
        return text

    tail = (tail or "").strip()
    if not tail:
        return _truncate_clean(text, max_chars)

    if len(tail) >= max_chars:
        return _truncate_clean(tail, max_chars)

    separator = "\n\n"
    head_budget = max_chars - len(tail) - len(separator)
    if head_budget <= 0:
        return _truncate_clean(tail, max_chars)

    head = _truncate_clean(text, head_budget)
    if not head:
        return _truncate_clean(tail, max_chars)

    combined = f"{head}{separator}{tail}".strip()
    if len(combined) <= max_chars:
        return combined

    adjusted_head_budget = max(0, head_budget - (len(combined) - max_chars))
    head = _truncate_clean(text, adjusted_head_budget)
    combined = f"{head}{separator}{tail}".strip() if head else tail
    if len(combined) <= max_chars:
        return combined

    return _truncate_clean(combined, max_chars)


def _fallback_post(
    user_query: str,
    linkedin_brief: Mapping[str, Any],
    hashtags: List[str],
) -> str:
    objective = str(
        linkedin_brief.get("objective", "Deliver practical insight")
    ).strip()
    objective = re.sub(
        r"support\s+'[^']+'\s+for:\s*",
        "Requested deliverable: ",
        objective,
        flags=re.IGNORECASE,
    ).strip()
    audience = str(linkedin_brief.get("audience", "operators and leaders")).strip()
    angle = (
        str(linkedin_brief.get("angle", "execution-focused perspective")).strip()
        or "execution-focused perspective"
    )
    topic_tokens = re.findall(r"[A-Za-z0-9]+", user_query or "")
    topic_summary = " ".join(topic_tokens[:8]).strip() or "requested topic"
    hashtags_line = " ".join(hashtags[:4]).strip()

    lines = [
        "## Fallback LinkedIn Outline",
        "",
        (
            "Text generation was unavailable, so this is a limited fallback "
            "structure based on retrieved research."
        ),
        "",
        f"- Topic: {topic_summary}",
        f"- Audience: {audience}",
        f"- Suggested angle: {angle}",
        (
            "- Suggested CTA: Ask the audience what criteria matter most when "
            "evaluating this topic."
        ),
    ]
    if objective:
        lines.append(f"- {objective}")
    if hashtags_line:
        lines.append(f"- Placeholder hashtags: {hashtags_line}")
    lines.append(
        (
            "- Next step: Regenerate this LinkedIn draft when provider availability "
            "returns."
        )
    )
    return "\n".join(lines).strip()


def _compose_final_post(
    body: str,
    hook: str,
    cta: str,
    hashtags: List[str],
    user_query: str,
    linkedin_brief: Mapping[str, Any],
    *,
    fallback_generated: bool = False,
) -> str:
    if fallback_generated:
        return _fallback_post(
            user_query=user_query,
            linkedin_brief=linkedin_brief,
            hashtags=hashtags,
        )

    text = body.strip()
    if not text:
        text = _fallback_post(
            user_query=user_query,
            linkedin_brief=linkedin_brief,
            hashtags=hashtags,
        )

    if hook not in text[:240]:
        text = f"{hook}\n\n{text}".strip()

    if cta not in text:
        text = f"{text}\n\n{cta}".strip()

    hashtags_line = " ".join(hashtags)
    if hashtags_line and hashtags_line not in text:
        text = f"{text}\n{hashtags_line}".strip()

    # Keep required CTA + hashtags when trimming (when feasible).
    tail = cta
    if hashtags_line:
        tail = f"{cta}\n{hashtags_line}"
    if len(text) > _MAX_LINKEDIN_CHARS:
        text = _truncate_with_tail(text=text, tail=tail, max_chars=_MAX_LINKEDIN_CHARS)

    # Deterministically expand if still too short.
    if len(text) < _MIN_LINKEDIN_CHARS:
        topic = (
            str(linkedin_brief.get("angle", "")).strip()
            or "AI workflow execution"
        )
        objective = str(
            linkedin_brief.get("objective", "operational consistency")
        ).strip()
        expansion = (
            " Practical detail: align your team around one weekly priority tied to "
            f"{topic}. Then turn that into a repeatable operating step, and "
            f"evaluate outcomes against {objective.lower()}."
        )
        while len(text) < _MIN_LINKEDIN_CHARS:
            text = f"{text}\n\n{expansion}".strip()

    if len(text) > _MAX_LINKEDIN_CHARS:
        text = _truncate_clean(text, _MAX_LINKEDIN_CHARS)

    return text


def _append_budget_error(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors = deepcopy(_safe_list(state.get("errors", [])))
    errors.append(
        {
            "agent": "linkedin_writer",
            "type": "budget_exceeded",
            "message": (
                "LinkedIn generation used deterministic fallback due to token "
                "budget limits."
            ),
            "recoverable": True,
        }
    )
    return errors


def _append_text_provider_degraded_error(
    state: Dict[str, Any],
    *,
    reason: str,
) -> List[Dict[str, Any]]:
    errors = deepcopy(_safe_list(state.get("errors", [])))
    errors.append(
        {
            "agent": "linkedin_writer",
            "type": "text_generation_degraded",
            "code": reason or "unknown_provider_error",
            "message": _FALLBACK_PROVIDER_WARNING,
            "recoverable": True,
        }
    )
    return errors


def _response_total_tokens(llm_response: Mapping[str, Any]) -> int:
    usage = _safe_dict(llm_response.get("usage", {}))
    total = usage.get("total_tokens", 0)
    if isinstance(total, bool):
        return 0
    if isinstance(total, int):
        return max(0, total)
    if isinstance(total, float):
        return max(0, int(total))
    return 0


def linkedin_writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a LinkedIn draft from the strategist brief."""
    user_query = str(state.get("user_query", "")).strip()
    content_brief = _safe_dict(state.get("content_brief", {}))
    linkedin_brief = _safe_dict(content_brief.get("linkedin", {}))

    content_drafts = _safe_dict(state.get("content_drafts", {}))
    existing_draft = _safe_dict(content_drafts.get("linkedin", {}))
    current_version = int(existing_draft.get("version", 0))
    next_version = current_version + 1

    retry_feedback_state = _safe_dict(state.get("retry_feedback", {}))
    raw_feedback = retry_feedback_state.get("linkedin", [])
    retry_feedback = [
        str(item).strip() for item in _safe_list(raw_feedback) if str(item).strip()
    ]
    draft_status = _safe_dict(state.get("draft_status", {}))

    cost_controls = normalize_cost_controls(_safe_dict(state.get("cost_controls", {})))
    errors = deepcopy(_safe_list(state.get("errors", [])))

    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        hook = _extract_hook("", user_query)
        cta = _extract_cta("")
        hashtags = _fallback_hashtags(user_query)
        final_body = _compose_final_post(
            body="",
            hook=hook,
            cta=cta,
            hashtags=hashtags,
            user_query=user_query,
            linkedin_brief=linkedin_brief,
            fallback_generated=True,
        )
        linkedin_update = {
            **existing_draft,
            "body": final_body,
            "version": next_version,
            "character_count": len(final_body),
            "hook": _extract_hook(final_body, user_query),
            "cta": _extract_cta(final_body),
            "hashtags": _extract_hashtags(final_body, user_query),
            "model_used": "budget_fallback",
            "fallback_generated": True,
            "degraded_generation": True,
            "generation_status": "fallback_degraded",
            "provider_status": "degraded",
            "provider_failure_reason": "quota_exceeded",
            "real_generation_succeeded": False,
            "generation_tokens": 0,
        }
        return {
            "content_drafts": {"linkedin": linkedin_update},
            "draft_status": {
                **draft_status,
                "linkedin": "complete",
            },
            "cost_controls": cost_controls,
            "errors": _append_budget_error(state),
            "status_messages": [_FALLBACK_PROVIDER_WARNING],
        }

    model = preferred_text_model(cost_controls)
    prompt = _build_prompt(
        user_query=user_query,
        linkedin_brief=linkedin_brief,
        retry_feedback=retry_feedback,
    )
    first_response = _safe_dict(
        generate_text(
            prompt=prompt,
            agent_key="linkedin_writer",
            model=model,
        )
    )
    cost_controls = apply_text_tokens(cost_controls, first_response)
    body = str(first_response.get("output", "")).strip()
    fallback_generated = bool(first_response.get("degraded", False))
    provider_failure_reason = str(
        _safe_dict(first_response.get("error", {})).get("code", "")
    ).strip().lower()
    if fallback_generated:
        body = ""

    if (
        not fallback_generated
        and len(body) < _MIN_LINKEDIN_CHARS
        and not token_budget_exceeded(cost_controls)
    ):
        retry_model = preferred_text_model(cost_controls)
        retry_prompt = _build_prompt(
            user_query=user_query,
            linkedin_brief=linkedin_brief,
            retry_feedback=retry_feedback,
            previous_char_count=len(body),
        )
        retry_response = _safe_dict(
            generate_text(
                prompt=retry_prompt,
                agent_key="linkedin_writer",
                model=retry_model,
            )
        )
        cost_controls = apply_text_tokens(cost_controls, retry_response)
        retry_body = str(retry_response.get("output", "")).strip()
        retry_degraded = bool(retry_response.get("degraded", False))
        if retry_degraded:
            fallback_generated = True
            body = ""
            provider_failure_reason = (
                str(_safe_dict(retry_response.get("error", {})).get("code", ""))
                .strip()
                .lower()
            ) or provider_failure_reason
        elif len(retry_body) >= len(body):
            body = retry_body
            model = retry_model
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
    if not body:
        fallback_generated = True

    hook = _extract_hook(body, user_query)
    cta = _extract_cta(body)
    hashtags = _extract_hashtags(body, user_query)

    final_body = _compose_final_post(
        body=body,
        hook=hook,
        cta=cta,
        hashtags=hashtags,
        user_query=user_query,
        linkedin_brief=linkedin_brief,
        fallback_generated=fallback_generated,
    )

    # Re-extract after composition to keep metadata aligned with final body.
    hook = _extract_hook(final_body, user_query)
    cta = _extract_cta(final_body)
    hashtags = _extract_hashtags(final_body, user_query)

    linkedin_update = {
        **existing_draft,
        "body": final_body,
        "version": next_version,
        "character_count": len(final_body),
        "hook": hook,
        "cta": cta,
        "hashtags": hashtags,
        "model_used": (
            str(first_response.get("model", "")).strip()
            or ("deterministic_fallback" if fallback_generated else model)
        ),
        "fallback_generated": fallback_generated,
        "degraded_generation": fallback_generated,
        "generation_status": (
            "fallback_degraded" if fallback_generated else "generated"
        ),
        "provider_status": "degraded" if fallback_generated else "ok",
        "provider_failure_reason": (
            provider_failure_reason if fallback_generated else ""
        ),
        "real_generation_succeeded": not fallback_generated,
        "generation_tokens": _response_total_tokens(first_response),
    }

    updates: Dict[str, Any] = {
        "content_drafts": {"linkedin": linkedin_update},
        "draft_status": {
            **draft_status,
            "linkedin": "complete",
        },
        "cost_controls": {
            **cost_controls,
        },
    }
    if errors:
        updates["errors"] = errors
    if fallback_generated:
        updates["errors"] = _append_text_provider_degraded_error(
            state,
            reason=provider_failure_reason or "unknown_provider_error",
        )
        updates["status_messages"] = [_FALLBACK_PROVIDER_WARNING]
    return updates
