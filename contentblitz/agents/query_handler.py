"""Query handler node implementation for Phase 1 scaffold.

This node performs lightweight query classification and returns only state
updates (patch semantics), not a full state object.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from contentblitz.core.cost_controls import (
    apply_text_tokens,
    normalize_cost_controls,
    preferred_text_model,
    token_budget_exceeded,
)
from contentblitz.safety.prompt_injection import analyze_prompt_injection
from contentblitz.tools.text import generate_text

_ALLOWED_OUTPUTS = {"blog", "linkedin", "image", "research"}
_DEFAULT_CLARIFICATION_MESSAGE = (
    "Could you clarify your goal, target audience, and desired output format?"
)
_INJECTION_WARNING_MESSAGE = (
    "Suspicious instruction patterns were detected and neutralized."
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_UNSAFE_INJECTION_TOKENS = {
    "ignore",
    "all",
    "any",
    "previous",
    "instruction",
    "instructions",
    "reveal",
    "show",
    "display",
    "print",
    "dump",
    "output",
    "expose",
    "system",
    "prompt",
    "prompts",
    "developer",
    "message",
    "messages",
    "hidden",
    "internal",
    "api",
    "key",
    "keys",
    "secret",
    "secrets",
    "environment",
    "env",
    "variable",
    "variables",
    "vars",
    "bypass",
    "disable",
    "guardrail",
    "guardrails",
    "safety",
    "protection",
    "protections",
    "safeguard",
    "safeguards",
    "and",
    "or",
    "then",
    "also",
}


def _normalize_outputs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    ordered = ["blog", "linkedin", "image", "research"]
    found = []
    for key in ordered:
        if any(str(item).strip().lower() == key for item in value):
            found.append(key)
    return found


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _with_injection_warning(state: Dict[str, Any]) -> list[str]:
    existing = [
        str(item).strip()
        for item in _safe_list(state.get("status_messages", []))
        if str(item).strip()
    ]
    if _INJECTION_WARNING_MESSAGE not in existing:
        existing.append(_INJECTION_WARNING_MESSAGE)
    return existing


def _injection_updates(
    state: Dict[str, Any],
    *,
    detected: bool,
    signals: list[str],
    sanitized_query: str,
) -> Dict[str, Any]:
    if not detected:
        return {}
    return {
        "prompt_injection_detected": True,
        "prompt_injection_signals": list(signals),
        "sanitized_user_query": sanitized_query,
        "status_messages": _with_injection_warning(state),
    }


def _has_safe_prompt_intent(query: str) -> bool:
    tokens = [token for token in _WORD_RE.findall(str(query).lower()) if token]
    if not tokens:
        return False
    return any(token not in _UNSAFE_INJECTION_TOKENS for token in tokens)


# TODO(intent-classification):
# Refine deterministic output classification so explicit LinkedIn-only
# requests do not automatically include blog generation unless requested.
def _parse_llm_classification(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = response.get("output", "")
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        return None

    if not isinstance(payload, dict):
        return None

    requested_outputs = _normalize_outputs(payload.get("requested_outputs"))
    clarification_needed = bool(payload.get("clarification_needed", False))
    clarification_message = payload.get("clarification_message")

    if not requested_outputs and not clarification_needed:
        return None

    intent = str(payload.get("intent", "")).strip().lower() or "general"
    research_required = bool(payload.get("research_required", False))
    export_requested = bool(payload.get("export_requested", False))

    if requested_outputs == ["image"]:
        research_required = False
    elif requested_outputs and requested_outputs != ["image"]:
        research_required = bool(payload.get("research_required", True))

    if clarification_needed and not isinstance(clarification_message, str):
        clarification_message = _DEFAULT_CLARIFICATION_MESSAGE
    if not clarification_needed:
        clarification_message = None

    return {
        "intent": intent,
        "requested_outputs": requested_outputs,
        "research_required": research_required,
        "clarification_needed": clarification_needed,
        "clarification_message": clarification_message,
        "export_requested": export_requested,
    }


def _deterministic_fallback(query: str) -> Dict[str, Any]:
    q = query.strip().lower()
    words = [w for w in q.split() if w]

    export_requested = any(token in q for token in ("export", "pdf", "download"))
    image_requested = any(
        token in q for token in ("image", "poster", "illustration", "graphic", "visual")
    )
    research_requested = any(
        token in q
        for token in ("research", "analyze", "analysis", "investigate", "sources")
    )
    blog_requested = any(token in q for token in ("blog", "article", "post"))
    linkedin_requested = "linkedin" in q

    vague_query = (
        not q
        or len(words) <= 2
        or q in {"help", "not sure", "something", "anything"}
        or (
            "help" in q
            and not any(
                (
                    image_requested,
                    research_requested,
                    blog_requested,
                    linkedin_requested,
                )
            )
        )
    )

    if vague_query:
        return {
            "intent": "clarification",
            "requested_outputs": [],
            "research_required": False,
            "clarification_needed": True,
            "clarification_message": _DEFAULT_CLARIFICATION_MESSAGE,
            "export_requested": export_requested,
        }

    requested_outputs: list[str] = []
    if blog_requested:
        requested_outputs.append("blog")
    if linkedin_requested:
        requested_outputs.append("linkedin")
    if image_requested:
        requested_outputs.append("image")
    if research_requested and not requested_outputs:
        requested_outputs.append("research")
    if not requested_outputs:
        requested_outputs = ["blog"]

    if "research" in requested_outputs and requested_outputs != ["research"]:
        requested_outputs = [item for item in requested_outputs if item != "research"]

    if any(item not in _ALLOWED_OUTPUTS for item in requested_outputs):
        requested_outputs = ["blog"]

    research_required = requested_outputs != ["image"]
    if requested_outputs == ["research"]:
        research_required = True

    if requested_outputs == ["image"]:
        intent = "image_generation"
    elif requested_outputs == ["research"]:
        intent = "research"
    else:
        intent = "content_creation"

    return {
        "intent": intent,
        "requested_outputs": requested_outputs,
        "research_required": research_required,
        "clarification_needed": False,
        "clarification_message": None,
        "export_requested": export_requested,
    }


def _determine_routing_decision(updates: Dict[str, Any]) -> str:
    if updates.get("clarification_needed"):
        return "clarification_node"
    outputs = updates.get("requested_outputs", [])
    if outputs == ["image"]:
        return "image_agent_node"
    if updates.get("research_required") or "research" in outputs:
        return "research_agent_node"
    return "content_strategist_node"


def _with_lifecycle_fields(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Attach stable lifecycle defaults for Phase 2 placeholder execution."""
    merged = dict(updates)
    merged["workflow_status"] = "routing_complete"
    merged["final_response"] = None
    return merged


def _append_budget_error(state: Dict[str, Any], message: str) -> list[dict[str, Any]]:
    existing_errors = state.get("errors", [])
    if not isinstance(existing_errors, list):
        existing_errors = []
    return [
        *existing_errors,
        {
            "node": "query_handler_node",
            "type": "budget_exceeded",
            "message": message,
        },
    ]


def query_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify the user query and return state updates only.

    This function never performs real API calls in Phase 1; generate_text is a
    local/mocked dependency.
    """
    cost_controls = normalize_cost_controls(_safe_dict(state.get("cost_controls", {})))
    budget_exceeded = bool(cost_controls.get("budget_exceeded", False))

    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        return _with_lifecycle_fields(
            {
                "cost_controls": cost_controls,
                "errors": _append_budget_error(
                    state,
                    "Session token budget exceeded before query classification.",
                ),
                "routing_decision": "error_handler_node",
            }
        )

    if budget_exceeded:
        return _with_lifecycle_fields(
            {
                "cost_controls": cost_controls,
                "errors": _append_budget_error(
                    state,
                    "Session budget exceeded before query classification.",
                ),
                "routing_decision": "error_handler_node",
            }
        )

    existing_errors = state.get("errors", [])
    if isinstance(existing_errors, list) and len(existing_errors) > 0:
        return _with_lifecycle_fields({"routing_decision": "error_handler_node"})

    query = str(state.get("user_query", "")).strip()
    injection_result = analyze_prompt_injection(query)
    effective_query = query
    injection_updates: Dict[str, Any] = {}
    if injection_result.detected:
        effective_query = injection_result.sanitized_query
        injection_updates = _injection_updates(
            state,
            detected=True,
            signals=injection_result.signals,
            sanitized_query=effective_query,
        )
        if effective_query and effective_query != query:
            injection_updates["user_query"] = effective_query

    if (
        query
        and injection_result.detected
        and not _has_safe_prompt_intent(effective_query)
    ):
        injection_updates["sanitized_user_query"] = ""
        injection_updates.pop("user_query", None)
        clarification_updates = {
            "intent": "clarification",
            "requested_outputs": [],
            "research_required": False,
            "clarification_needed": True,
            "clarification_message": _DEFAULT_CLARIFICATION_MESSAGE,
            "export_requested": bool(state.get("export_requested", False)),
            "routing_decision": "clarification_node",
            "cost_controls": cost_controls,
        }
        return _with_lifecycle_fields({**clarification_updates, **injection_updates})

    preset_outputs = _normalize_outputs(state.get("requested_outputs", []))
    if (
        effective_query
        and preset_outputs == ["image"]
        and not bool(state.get("clarification_needed", False))
    ):
        image_only = {
            "intent": str(state.get("intent", "")).strip().lower()
            or "image_generation",
            "requested_outputs": ["image"],
            "research_required": False,
            "clarification_needed": False,
            "clarification_message": None,
            "export_requested": bool(state.get("export_requested", False)),
        }
        image_only["routing_decision"] = _determine_routing_decision(image_only)
        image_only["cost_controls"] = cost_controls
        return _with_lifecycle_fields({**image_only, **injection_updates})

    if not effective_query and (
        preset_outputs
        or bool(state.get("clarification_needed", False))
        or bool(state.get("research_required", False))
        or bool(state.get("export_requested", False))
    ):
        preclassified = {
            "intent": str(state.get("intent", "")).strip().lower() or "preclassified",
            "requested_outputs": preset_outputs,
            "research_required": bool(state.get("research_required", False)),
            "clarification_needed": bool(state.get("clarification_needed", False)),
            "clarification_message": state.get("clarification_message"),
            "export_requested": bool(state.get("export_requested", False)),
        }
        if preclassified["requested_outputs"] == ["image"]:
            preclassified["research_required"] = False
        if preclassified["clarification_needed"] and not isinstance(
            preclassified["clarification_message"], str
        ):
            preclassified["clarification_message"] = _DEFAULT_CLARIFICATION_MESSAGE
        if not preclassified["clarification_needed"]:
            preclassified["clarification_message"] = None
        preclassified["routing_decision"] = _determine_routing_decision(preclassified)
        preclassified["cost_controls"] = cost_controls
        return _with_lifecycle_fields({**preclassified, **injection_updates})

    prompt = (
        "Classify the user request for ContentBlitz. "
        "Return strict JSON with keys: intent, requested_outputs, research_required, "
        "clarification_needed, clarification_message, export_requested.\n\n"
        f"User query: {effective_query}"
    )
    model = preferred_text_model(cost_controls)
    llm_response = generate_text(
        prompt=prompt,
        agent_key="query_handler",
        model=model,
    )
    cost_controls = apply_text_tokens(cost_controls, llm_response)
    if token_budget_exceeded(cost_controls):
        cost_controls["budget_exceeded"] = True
        return _with_lifecycle_fields(
            {
                "cost_controls": cost_controls,
                "errors": _append_budget_error(
                    state,
                    "Session token budget exceeded during query classification.",
                ),
                "routing_decision": "error_handler_node",
            }
        )
    classified = _parse_llm_classification(llm_response)
    if classified is None:
        classified = _deterministic_fallback(effective_query)

    classified["routing_decision"] = _determine_routing_decision(classified)
    classified["cost_controls"] = cost_controls
    return _with_lifecycle_fields({**classified, **injection_updates})
