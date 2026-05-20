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
_OBSERVABILITY_STATUSES = {"enabled", "disabled", "degraded"}
_OBSERVABILITY_STATUS_LABELS = {
    "enabled": "Enabled",
    "disabled": "Disabled",
    "degraded": "Degraded",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _normalize_format_list(value: Any) -> list[str]:
    normalized: list[str] = []
    for item in _safe_list(value):
        token = str(item).strip().lower()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _failed_export_formats(export_metadata: Mapping[str, Any]) -> list[str]:
    explicit_failed = _normalize_format_list(
        export_metadata.get("failed_export_formats")
    )
    if explicit_failed:
        return explicit_failed
    status = _safe_dict(export_metadata.get("export_status", {}))
    return [
        str(fmt).strip().lower()
        for fmt, value in status.items()
        if str(fmt).strip() and str(value).strip().lower() == "failed"
    ]


def _is_warning_export_log_entry(entry: Mapping[str, Any]) -> bool:
    code = str(entry.get("code", "")).strip().lower()
    if code.endswith("_warning") or code == "warning":
        return True
    message = str(entry.get("message", "")).strip().lower()
    return "warning" in message and "failed" not in message


def _export_failure_count(export_metadata: Mapping[str, Any]) -> int:
    explicit = _safe_int(export_metadata.get("export_error_count"), default=-1)
    if explicit >= 0:
        return explicit
    failed_formats = _failed_export_formats(export_metadata)
    if failed_formats:
        return len(failed_formats)
    return sum(
        1
        for item in _safe_list(export_metadata.get("error_log", []))
        if isinstance(item, Mapping) and not _is_warning_export_log_entry(item)
    )


def _export_warning_count(export_metadata: Mapping[str, Any]) -> int:
    explicit = _safe_int(export_metadata.get("export_warning_count"), default=-1)
    if explicit >= 0:
        return explicit
    return sum(
        1
        for item in _safe_list(export_metadata.get("error_log", []))
        if isinstance(item, Mapping) and _is_warning_export_log_entry(item)
    )


def normalize_observability_status(status: Any) -> str:
    """Normalize observability status to a safe bounded set."""
    normalized = str(status).strip().lower()
    if normalized in _OBSERVABILITY_STATUSES:
        return normalized
    return "disabled"


def observability_status_label(status: Any) -> str:
    """Return a display-safe label for observability status."""
    normalized = normalize_observability_status(status)
    return _OBSERVABILITY_STATUS_LABELS.get(normalized, "Disabled")


def _dedupe_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for message in messages:
        cleaned = str(message).strip()
        if not cleaned or cleaned.lower() in {"none", "null"} or cleaned in seen:
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
        if str(error.get("agent", "")).strip().lower() == "image_agent" and bool(
            error.get("recoverable", False)
        ):
            return True
    return False


def _is_fallback_draft(draft: Mapping[str, Any]) -> bool:
    if bool(draft.get("fallback_generated", False)):
        return True
    if bool(draft.get("degraded_generation", False)):
        return True
    generation_status = str(draft.get("generation_status", "")).strip().lower()
    if generation_status in {"fallback_degraded", "fallback_generated"}:
        return True
    provider_status = str(draft.get("provider_status", "")).strip().lower()
    return provider_status == "degraded"


def _has_text_generation_degradation(state: Mapping[str, Any]) -> bool:
    content_drafts = _safe_dict(state.get("content_drafts", {}))
    for channel in ("blog", "linkedin"):
        draft = _safe_dict(content_drafts.get(channel, {}))
        if _is_fallback_draft(draft):
            return True
    return False


def _has_terminal_failure(
    state: Mapping[str, Any], node_statuses: Mapping[str, str]
) -> bool:
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
    clarification_required: bool = False,
) -> str:
    """Summarize workflow status for display without leaking internals."""
    statuses = {
        node: normalize_progress_status(status)
        for node, status in dict(node_statuses).items()
    }
    status_values = list(statuses.values())

    if any(status == "failed" for status in status_values):
        return "failed"

    normalized_workflow_status = str(workflow_status).strip().lower()
    requires_clarification = bool(clarification_required) or (
        normalized_workflow_status == "awaiting_clarification"
    )
    if requires_clarification:
        return "awaiting_clarification"

    if any(status == "running" for status in status_values):
        return "running"
    if any(status == "degraded" for status in status_values):
        return "partial_success"

    if normalized_workflow_status in {"partial_success", "completed_with_warnings"}:
        return "partial_success"
    if normalized_workflow_status in {"success", "research_complete"}:
        return "success"
    if all(status == "pending" for status in status_values):
        return "pending"
    if any(status in _TERMINAL_STATUSES for status in status_values):
        return "success"
    return "unknown"


def apply_optional_node_skips(
    *,
    state: Mapping[str, Any],
    node_statuses: Mapping[str, str],
) -> dict[str, str]:
    """
    Mark optional nodes as skipped when they were not requested and did not run.

    This keeps UI node status truthful without mutating orchestration state.
    """
    updated = {
        key: normalize_progress_status(value)
        for key, value in dict(node_statuses).items()
    }
    outputs = {
        str(item).strip().lower()
        for item in _safe_list(state.get("requested_outputs", []))
        if str(item).strip()
    }
    export_metadata = _safe_dict(state.get("export_metadata", {}))
    formats_requested = _safe_list(export_metadata.get("formats_requested", []))
    export_requested = bool(state.get("export_requested", False)) or bool(
        formats_requested
    )

    def _mark_skipped(node_name: str) -> None:
        current = updated.get(node_name, "pending")
        if current in {"completed", "degraded", "failed", "running"}:
            return
        updated[node_name] = "skipped"

    if "blog" not in outputs:
        _mark_skipped("blog_writer_node")
    if "linkedin" not in outputs:
        _mark_skipped("linkedin_writer_node")
    if "image" not in outputs:
        _mark_skipped("image_agent_node")
    if "research" not in outputs and not bool(state.get("research_required", False)):
        _mark_skipped("research_agent_node")

    if not export_requested:
        updated["export_node"] = "skipped"

    return updated


def workflow_requires_clarification(
    *,
    state: Mapping[str, Any],
    node_statuses: Mapping[str, str],
) -> bool:
    if bool(state.get("clarification_needed", False)):
        return True
    if (
        str(state.get("workflow_status", "")).strip().lower()
        == "awaiting_clarification"
    ):
        return True
    clarification_status = normalize_progress_status(
        node_statuses.get("clarification_node", "pending")
    )
    if clarification_status in {"running", "completed"}:
        return True
    return False


def build_status_messages(
    *,
    state: Mapping[str, Any],
    node_statuses: Mapping[str, str],
) -> list[str]:
    """Build high-level user-safe status lines for workflow UI rendering."""
    normalized_statuses = apply_optional_node_skips(
        state=state,
        node_statuses=node_statuses,
    )
    clarification_required = workflow_requires_clarification(
        state=state,
        node_statuses=normalized_statuses,
    )
    summary = summarize_workflow_status(
        normalized_statuses,
        workflow_status=str(state.get("workflow_status", "")),
        clarification_required=clarification_required,
    )
    messages: list[str] = []

    if summary == "running":
        messages.append("Workflow is currently running.")
    elif summary == "failed":
        messages.append("Workflow ended with a terminal failure.")
    elif summary == "awaiting_clarification":
        messages.append("Workflow paused awaiting clarification.")
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
            (
                "Image generation encountered a recoverable issue. "
                "Text outputs remain available."
            )
        )

    if _has_text_generation_degradation(state) or _has_recoverable_image_failure(state):
        messages.append(
            "OpenAI provider unavailable or quota-limited. "
            "ContentBlitz generated limited fallback outputs."
        )

    export_metadata = _safe_dict(state.get("export_metadata", {}))
    export_failures = _export_failure_count(export_metadata)
    export_warnings = _export_warning_count(export_metadata)
    final_response = str(state.get("final_response", "")).strip()
    if export_failures > 0 and final_response:
        messages.append(
            "One or more exports failed, but the final response is still available."
        )
    elif export_warnings > 0:
        messages.append("Export completed with non-blocking warnings.")

    cost_controls = _safe_dict(state.get("cost_controls", {}))
    if bool(cost_controls.get("budget_exceeded", False)):
        messages.append("Budget limits were reached during this workflow.")
    retries = int(cost_controls.get("total_retries_used_this_session", 0))
    if retries > 0:
        messages.append(f"Retries used this session: {retries}.")

    if _has_terminal_failure(state, normalized_statuses):
        messages.append("Try refining your prompt and rerunning the workflow.")

    return _dedupe_messages(messages)
