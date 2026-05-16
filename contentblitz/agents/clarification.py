"""Clarification node implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from contentblitz.tools.text import generate_text

_DEFAULT_CLARIFICATION_MESSAGE = (
    "Could you clarify your goal, target audience, and desired output format?"
)


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _extract_message(response: Mapping[str, Any]) -> str:
    raw = response.get("output", "")
    if isinstance(raw, str):
        message = raw.strip()
    elif isinstance(raw, Mapping):
        for key in ("clarification_message", "message", "text", "question", "output"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break
        else:
            message = ""
    else:
        message = ""

    if message and "?" not in message:
        message = f"{message}?"
    return message


def _resolve_clarification_message(state: Dict[str, Any]) -> str:
    existing = state.get("clarification_message")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    query = str(state.get("user_query", "")).strip()
    prompt = (
        "Write one short clarification question for an ambiguous content request. "
        "Ask for concrete goals and desired output type.\n\n"
        f"User query: {query}"
    )
    try:
        response = generate_text(prompt=prompt, agent_key="query_handler")
        message = _extract_message(response)
    except Exception:
        message = ""

    return message or _DEFAULT_CLARIFICATION_MESSAGE


def clarification_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a clarification response and stop execution path at END.

    This node is deterministic and does not call external APIs.
    """
    query = str(state.get("user_query", "")).strip()
    history = deepcopy(_safe_list(state.get("conversation_history", [])))
    history.append({"role": "user", "content": query})

    message = _resolve_clarification_message(state)

    return {
        "clarification_message": message,
        "conversation_history": history,
        "final_response": message,
        "workflow_status": "awaiting_clarification",
        "assembled_outputs": {},
        "export_outputs": {},
    }


__all__ = ["clarification_node"]
