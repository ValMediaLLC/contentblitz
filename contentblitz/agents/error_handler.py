"""Error handler node implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping

_GENERIC_FATAL_MESSAGE = (
    "We hit an unexpected error while processing your request. "
    "Please try again with a more specific prompt."
)
_GENERIC_WARNING_MESSAGE = (
    "Your request completed with warnings. Some non-blocking issues occurred."
)
_FATAL_INTERNAL_ERROR_TYPES = {
    "unexpected_exception",
    "provider_failure",
    "validation_exception",
    "assembly_failed",
    "budget_exceeded",
}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _normalize_errors(raw_errors: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in raw_errors:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
        else:
            normalized.append(
                {
                    "agent": "unknown",
                    "type": "unknown_error",
                    "message": "Unexpected non-structured error.",
                    "recoverable": False,
                }
            )
    return normalized


def _is_fatal_error(error: Mapping[str, Any]) -> bool:
    if not bool(error.get("recoverable", False)):
        return True
    err_type = str(error.get("type", "")).strip().lower()
    if err_type in _FATAL_INTERNAL_ERROR_TYPES:
        return True
    return False


def _is_fatal_state(errors: List[Dict[str, Any]]) -> bool:
    if not errors:
        return False
    return any(_is_fatal_error(item) for item in errors)


def _terminal_error_entry(fatal: bool) -> Dict[str, Any]:
    if fatal:
        return {
            "agent": "error_handler",
            "type": "terminal_error",
            "message": "Fatal error response generated.",
            "recoverable": False,
        }
    return {
        "agent": "error_handler",
        "type": "terminal_warning",
        "message": "Recoverable warning response generated.",
        "recoverable": True,
    }


def error_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a safe user-facing response for error states.

    This node does not call external APIs and avoids exposing raw internal errors.
    """
    next_state = deepcopy(state)
    existing_errors = _normalize_errors(_safe_list(next_state.get("errors", [])))
    fatal = _is_fatal_state(existing_errors)

    if fatal:
        final_response = _GENERIC_FATAL_MESSAGE
        workflow_status = "failed"
    elif existing_errors:
        final_response = _GENERIC_WARNING_MESSAGE
        workflow_status = "completed_with_warnings"
    else:
        final_response = _GENERIC_FATAL_MESSAGE
        workflow_status = "error_handled"

    next_state["errors"] = [*existing_errors, _terminal_error_entry(fatal=fatal)]
    next_state["final_response"] = final_response
    next_state["workflow_status"] = workflow_status
    return next_state
