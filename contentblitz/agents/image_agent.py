"""Image agent node scaffold."""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any, Dict, List, Mapping

from contentblitz.core.cost_controls import (
    apply_text_tokens,
    image_cap_reached,
    normalize_cost_controls,
    preferred_text_model,
    token_budget_exceeded,
)
from contentblitz.tools.image import generate_image
from contentblitz.tools.text import generate_text

_IMAGE_FAILURE_MESSAGE = "No image assets returned."
_SAFE_IMAGE_PROVIDER_WARNING = (
    "Image generation encountered a recoverable issue. "
    "Text/research/export outputs remain available."
)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_base64_payload(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if not lowered:
        return False
    return lowered.startswith("data:image/") or "base64" in lowered


def _derive_visual_concept(state: Dict[str, Any]) -> str:
    content_brief = _safe_dict(state.get("content_brief", {}))
    image_brief = _safe_dict(content_brief.get("image", {}))

    preferred_fields = (
        "prompt",
        "prompt_focus",
        "visual_concept",
        "concept",
        "brief",
        "angle",
    )
    base = ""
    for field in preferred_fields:
        candidate = str(image_brief.get(field, "")).strip()
        if candidate:
            base = candidate
            break

    visual_direction = str(image_brief.get("visual_direction", "")).strip()
    style_hint = str(image_brief.get("style", "")).strip()

    if base:
        if visual_direction:
            return f"{base}. Visual direction: {visual_direction}."
        return base

    user_query = (
        str(state.get("user_query", "")).strip() or "Strategic marketing concept"
    )
    research_data = _safe_dict(state.get("research_data", {}))
    summary = (
        str(research_data.get("synthesized_summary", "")).strip()
        or str(research_data.get("summary", "")).strip()
    )
    keywords = [
        str(item).strip()
        for item in _safe_list(research_data.get("keywords", []))
        if str(item).strip()
    ][:3]

    concept = f"Create an image concept for: {user_query}."
    if summary:
        concept += f" Context: {summary}"
    if keywords:
        concept += f" Keywords: {', '.join(keywords)}."
    if visual_direction:
        concept += f" Visual direction: {visual_direction}."
    if style_hint:
        concept += f" Style: {style_hint}."
    return concept.strip()


def _enhance_prompt(
    concept: str, cost_controls: Mapping[str, Any]
) -> tuple[str, Dict[str, Any]]:
    prompt = (
        "Enhance this image generation prompt for clarity and visual detail. "
        "Return only the improved prompt.\n\n"
        f"Prompt: {concept}"
    )
    response = _safe_dict(
        generate_text(
            prompt=prompt,
            agent_key="image_agent",
            model=preferred_text_model(cost_controls, agent_key="image_agent"),
        )
    )
    enhanced = str(response.get("output", "")).strip()
    if enhanced:
        return enhanced, response
    return f"{concept} High detail, clean composition, and strong contrast.", response


def _sanitize_image_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    allowed_keys = (
        "url",
        "local_path",
        "id",
        "mime_type",
        "width",
        "height",
        "renderable",
        "revised_prompt",
    )
    sanitized: Dict[str, Any] = {}
    for key in allowed_keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if key in {"url", "local_path"} and _is_base64_payload(value):
            continue
        sanitized[key] = value
    return sanitized


def _normalize_provider(response: Mapping[str, Any]) -> str:
    provider = str(response.get("provider_used", "")).strip()
    if provider:
        return provider
    primary = str(response.get("provider_primary", "")).strip()
    if primary:
        return primary
    return "dall-e-3"


def _append_recoverable_image_error(
    state: Dict[str, Any],
    message: str,
    *,
    error_type: str = "image_generation_failed",
) -> List[Dict[str, Any]]:
    existing_errors = deepcopy(_safe_list(state.get("errors", [])))
    existing_errors.append(
        {
            "agent": "image_agent",
            "type": error_type,
            "message": message,
            "recoverable": True,
        }
    )
    return existing_errors


def _safe_image_error_payload(raw_error: Any) -> Dict[str, Any]:
    error = _safe_dict(raw_error)
    code = _safe_text(error.get("code")).lower() or "unknown_provider_error"
    message = _safe_text(error.get("message")) or _SAFE_IMAGE_PROVIDER_WARNING
    recoverable = bool(error.get("recoverable", True))
    return {
        "code": code,
        "message": message,
        "recoverable": recoverable,
    }


def _provider_perf_metrics(
    *,
    provider_latency_ms: int | None,
    provider_call_count: int,
) -> Dict[str, int]:
    if provider_call_count <= 0:
        return {}
    safe_latency = (
        0 if provider_latency_ms is None else max(0, int(provider_latency_ms))
    )
    return {
        "provider_call_count": provider_call_count,
        "provider_latency_ms": safe_latency,
    }


# TODO(provider):
# Replace recoverable image failure with a deterministic fallback image asset
# when provider-specific fallback asset policy is finalized.
def image_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate images via deterministic prompt enhancement + stateless tool calls."""
    cost_controls = normalize_cost_controls(_safe_dict(state.get("cost_controls", {})))
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        tool_outputs = deepcopy(_safe_dict(state.get("tool_outputs", {})))
        tool_outputs["image_agent"] = {
            "status": "skipped",
            "reason": "token_budget_exceeded",
            "attempted": False,
        }
        return {
            "tool_outputs": tool_outputs,
            "draft_status": {"image": "skipped"},
            "errors": _append_recoverable_image_error(
                state,
                "Image generation skipped because session token budget is exceeded.",
                error_type="budget_exceeded",
            ),
            "cost_controls": cost_controls,
        }

    if image_cap_reached(cost_controls):
        tool_outputs = deepcopy(_safe_dict(state.get("tool_outputs", {})))
        tool_outputs["image_agent"] = {
            "status": "skipped",
            "reason": "image_generation_cap_reached",
            "attempted": False,
        }
        return {
            "tool_outputs": tool_outputs,
            "draft_status": {"image": "skipped"},
            "errors": _append_recoverable_image_error(
                state,
                "Image generation skipped because the image generation cap "
                "was reached.",
                error_type="image_generation_cap_reached",
            ),
            "cost_controls": cost_controls,
        }

    concept = _derive_visual_concept(state)
    enhanced_prompt, enhancement_response = _enhance_prompt(concept, cost_controls)
    cost_controls = apply_text_tokens(cost_controls, enhancement_response)
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        tool_outputs = deepcopy(_safe_dict(state.get("tool_outputs", {})))
        tool_outputs["image_agent"] = {
            "status": "skipped",
            "reason": "token_budget_exceeded_after_prompt_enhancement",
            "attempted": False,
        }
        return {
            "tool_outputs": tool_outputs,
            "draft_status": {"image": "skipped"},
            "errors": _append_recoverable_image_error(
                state,
                "Image generation skipped because token budget was exhausted.",
                error_type="budget_exceeded",
            ),
            "cost_controls": cost_controls,
        }

    content_brief = _safe_dict(state.get("content_brief", {}))
    image_brief = _safe_dict(content_brief.get("image", {}))
    style = str(image_brief.get("style", "")).strip() or "default"

    image_prompts = deepcopy(_safe_list(state.get("image_prompts", [])))
    image_prompts.append(enhanced_prompt)

    image_outputs = deepcopy(_safe_list(state.get("image_outputs", [])))
    tool_outputs = deepcopy(_safe_dict(state.get("tool_outputs", {})))

    used = int(cost_controls.get("image_generations_used_this_session", 0))

    provider_latency_ms: int | None = None
    provider_call_count = 0
    try:
        provider_started_at = perf_counter()
        image_response = _safe_dict(generate_image(prompt=enhanced_prompt, style=style))
        provider_latency_ms = max(
            0,
            int((perf_counter() - provider_started_at) * 1000),
        )
        provider_call_count = 1
    except Exception:  # pragma: no cover - defensive path
        provider_latency_ms = max(
            0,
            int((perf_counter() - provider_started_at) * 1000),
        )
        provider_call_count = 1
        failure_message = _SAFE_IMAGE_PROVIDER_WARNING
        failure_payload = {
            "status": "failed",
            "recoverable": True,
            "error": {
                "code": "unknown_provider_error",
                "message": _SAFE_IMAGE_PROVIDER_WARNING,
                "recoverable": True,
            },
            "prompt": enhanced_prompt,
            "provider": "dall-e-3",
        }
        image_outputs.append(failure_payload)
        tool_outputs["image_agent"] = {
            "status": "failed",
            "recoverable": True,
            "reason": "unknown_provider_error",
            "attempted": True,
            "provider_status": "degraded",
            **_provider_perf_metrics(
                provider_latency_ms=provider_latency_ms,
                provider_call_count=provider_call_count,
            ),
        }
        return {
            "image_prompts": image_prompts,
            "image_outputs": image_outputs,
            "tool_outputs": tool_outputs,
            "draft_status": {"image": "failed"},
            "errors": _append_recoverable_image_error(state, failure_message),
            "cost_controls": cost_controls,
            "status_messages": [_SAFE_IMAGE_PROVIDER_WARNING],
        }

    provider = _normalize_provider(image_response)
    raw_images = _safe_list(image_response.get("images", []))

    sanitized_images: List[Dict[str, Any]] = []
    for raw_image in raw_images:
        sanitized = _sanitize_image_payload(raw_image)
        if not sanitized:
            continue
        sanitized_images.append(
            {
                "status": "success",
                "provider": provider,
                "prompt": enhanced_prompt,
                **sanitized,
            }
        )

    if not sanitized_images:
        error_payload = _safe_image_error_payload(image_response.get("error", {}))
        failure_reason = (
            _safe_text(error_payload.get("code")) or "empty_provider_response"
        )
        failure_payload = {
            "status": "failed",
            "recoverable": True,
            "error": error_payload,
            "prompt": enhanced_prompt,
            "provider": provider,
        }
        image_outputs.append(failure_payload)
        tool_outputs["image_agent"] = {
            "status": "failed",
            "recoverable": True,
            "reason": failure_reason,
            "attempted": True,
            "provider_status": "degraded",
            **_provider_perf_metrics(
                provider_latency_ms=provider_latency_ms,
                provider_call_count=provider_call_count,
            ),
        }
        return {
            "image_prompts": image_prompts,
            "image_outputs": image_outputs,
            "tool_outputs": tool_outputs,
            "draft_status": {"image": "failed"},
            "errors": _append_recoverable_image_error(
                state,
                _safe_text(error_payload.get("message"))
                or _SAFE_IMAGE_PROVIDER_WARNING,
            ),
            "cost_controls": cost_controls,
            "status_messages": [_SAFE_IMAGE_PROVIDER_WARNING],
        }

    renderable_count = 0
    for entry in sanitized_images:
        has_renderable_ref = bool(
            _safe_text(entry.get("url")) or _safe_text(entry.get("local_path"))
        )
        explicit_renderable = entry.get("renderable")
        is_renderable = (
            bool(explicit_renderable)
            if isinstance(explicit_renderable, bool)
            else has_renderable_ref
        )
        if is_renderable:
            entry["status"] = "success"
            entry["renderable"] = True
            renderable_count += 1
        else:
            entry["status"] = "degraded"
            entry["renderable"] = False

    image_outputs.extend(sanitized_images)
    cost_controls["image_generations_used_this_session"] = used + 1

    if renderable_count <= 0:
        non_renderable_reason = (
            "Image provider returned a non-renderable asset reference."
        )
        tool_outputs["image_agent"] = {
            "status": "degraded",
            "recoverable": True,
            "reason": non_renderable_reason,
            "attempted": True,
            "provider": provider,
            "provider_status": "degraded",
            "images_generated": len(sanitized_images),
            "renderable_images": renderable_count,
            **_provider_perf_metrics(
                provider_latency_ms=provider_latency_ms,
                provider_call_count=provider_call_count,
            ),
        }
        return {
            "image_prompts": image_prompts,
            "image_outputs": image_outputs,
            "tool_outputs": tool_outputs,
            "draft_status": {"image": "failed"},
            "errors": _append_recoverable_image_error(state, non_renderable_reason),
            "cost_controls": cost_controls,
            "status_messages": [_SAFE_IMAGE_PROVIDER_WARNING],
        }

    tool_outputs["image_agent"] = {
        "status": "success",
        "recoverable": False,
        "attempted": True,
        "provider": provider,
        "provider_status": "ok",
        "images_generated": len(sanitized_images),
        "renderable_images": renderable_count,
        **_provider_perf_metrics(
            provider_latency_ms=provider_latency_ms,
            provider_call_count=provider_call_count,
        ),
    }
    return {
        "image_prompts": image_prompts,
        "image_outputs": image_outputs,
        "tool_outputs": tool_outputs,
        "draft_status": {"image": "complete"},
        "cost_controls": cost_controls,
    }
