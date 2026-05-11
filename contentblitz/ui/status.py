"""Workflow status helpers for the UI layer."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from contentblitz.ui.error_display import normalize_errors_for_display
from contentblitz.ui.progress import (
    UIProgressEvent,
    normalize_progress_status,
    order_progress_events,
    validate_node_name,
)
from contentblitz.workflow.routing import AUTHORITATIVE_NODES

_TERMINAL_STATUSES = {"completed", "skipped", "degraded", "failed"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for message in messages:
        cleaned = str(message).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _has_recoverable_image_failure(state: Mapping[str, Any]) -> bool:
    image_outputs = _safe_list(state.get("image_outputs", []))
    for output in image_outputs:
        if not isinstance(output, Mapping):
            continue
        if str(output.get("status", "")).strip().lower() == "failed":
            return True
    errors = normalize_errors_for_display(_safe_list(state.get("errors", [])))
    for error in errors:
        if (
            str(error.get("agent", "")).strip().lower() == "image_agent"
            and bool(error.get("recoverable", False))
        ):
            return True
    return False


def _has_terminal_failure(state: Mapping[str, Any], node_statuses: Mapping[str, str]) -> bool:
    workflow_status = str(state.get("workflow_status", "")).strip().lower()
    if workflow_status in {"failed", "error", "error_handled"}:
        return True
    return any(status == "failed" for status in node_statuses.values())


def build_initial_node_statuses(
    *,
    node_names: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return a deterministic pending status map for authoritative nodes."""
    names = list(node_names) if node_names is not None else list(AUTHORITATIVE_NODES)
    statuses: dict[str, str] = {}
    for name in names:
        validated = validate_node_name(str(name))
        statuses[validated] = "pending"
    return statuses


def derive_node_statuses(
    events: Iterable[UIProgressEvent | Mapping[str, Any]],
    *,
    node_names: Iterable[str] | None = None,
) -> dict[str, str]:
    """Reduce progress events into a latest-status view per node."""
    statuses = build_initial_node_statuses(node_names=node_names)
    for event in order_progress_events(events):
        statuses[event.node_name] = normalize_progress_status(event.status)
    return statuses


def summarize_workflow_status(
    node_statuses: Mapping[str, str],
    *,
    workflow_status: str = "",
) -> str:
    """Summarize workflow status for display without leaking internals."""
    normalized_workflow_status = str(workflow_status).strip().lower()
    if normalized_workflow_status in {"failed", "error", "error_handled"}:
        return "failed"
    if normalized_workflow_status in {"partial_success", "completed_with_warnings"}:
        return "partial_success"
    if normalized_workflow_status in {"success", "awaiting_clarification", "research_complete"}:
        return "success"

    statuses = {
        node: normalize_progress_status(status)
        for node, status in dict(node_statuses).items()
    }
    status_values = list(statuses.values())
    if any(status == "failed" for status in status_values):
        return "failed"
    if any(status == "running" for status in status_values):
        return "running"
    if any(status == "degraded" for status in status_values):
        return "partial_success"
    if all(status == "pending" for status in status_values):
        return "pending"
    if any(status in _TERMINAL_STATUSES for status in status_values):
        return "completed"
    return "unknown"


def build_status_messages(
    *,
    state: Mapping[str, Any],
    node_statuses: Mapping[str, str],
) -> list[str]:
    """Build high-level user-safe status lines for workflow UI rendering."""
    normalized_statuses = {
        node: normalize_progress_status(status)
        for node, status in dict(node_statuses).items()
    }
    summary = summarize_workflow_status(
        normalized_statuses,
        workflow_status=str(state.get("workflow_status", "")),
    )
    messages: list[str] = []

    if summary == "running":
        messages.append("Workflow is currently running.")
    elif summary == "failed":
        messages.append("Workflow ended with a terminal failure.")
    elif summary == "partial_success":
        messages.append("Workflow completed with recoverable warnings.")
    elif summary in {"success", "completed"}:
        messages.append("Workflow completed successfully.")

    research_data = _safe_dict(state.get("research_data", {}))
    if bool(research_data.get("degraded", False)):
        messages.append(
            "Research results are degraded. Validate sources before publishing."
        )

    if _has_recoverable_image_failure(state):
        messages.append(
            "Image generation encountered a recoverable issue. Text outputs remain available."
        )

    export_metadata = _safe_dict(state.get("export_metadata", {}))
    export_errors = _safe_list(export_metadata.get("error_log", []))
    final_response = str(state.get("final_response", "")).strip()
    if export_errors and final_response:
        messages.append(
            "One or more exports failed, but the final response is still available."
        )

    cost_controls = _safe_dict(state.get("cost_controls", {}))
    if bool(cost_controls.get("budget_exceeded", False)):
        messages.append("Budget limits were reached during this workflow.")
    retries = int(cost_controls.get("total_retries_used_this_session", 0))
    if retries > 0:
        messages.append(f"Retries used this session: {retries}.")

    if _has_terminal_failure(state, normalized_statuses):
        messages.append("Try refining your prompt and rerunning the workflow.")

    return _dedupe_messages(messages)

