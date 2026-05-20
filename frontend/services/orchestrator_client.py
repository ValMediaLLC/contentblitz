"""Thin frontend adapter over ContentBlitz orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, Iterable, Iterator, List, Mapping

from contentblitz.state import create_initial_state
from contentblitz.ui.progress import (
    UIProgressEvent,
    create_progress_event,
    order_progress_events,
)
from contentblitz.workflow.graph import build_langgraph
from contentblitz.workflow.routing import (
    AUTHORITATIVE_NODE_SET,
    ERROR_HANDLER_NODE,
    EXPORT_NODE,
    IMAGE_AGENT_NODE,
    RESEARCH_AGENT_NODE,
)

_GRAPH = None


def _get_graph():
    """Build graph lazily to keep frontend startup side-effect free."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_langgraph()
    return _GRAPH


def _build_initial_state(
    *,
    user_query: str,
    requested_outputs: List[str] | None,
    export_requested: bool = False,
    export_formats: List[str] | None,
) -> Dict[str, Any]:
    safe_query = str(user_query).strip()
    safe_outputs = [
        str(item).strip() for item in (requested_outputs or []) if str(item).strip()
    ]
    safe_export_formats = [
        str(item).strip().lower()
        for item in (export_formats or [])
        if str(item).strip()
    ]

    initial_state_overrides: Dict[str, Any] = {"user_query": safe_query}
    if safe_outputs:
        initial_state_overrides["requested_outputs"] = safe_outputs
    if export_requested or safe_export_formats:
        initial_state_overrides["export_requested"] = bool(export_requested)
        initial_state_overrides["export_metadata"] = {
            "formats_requested": safe_export_formats if export_requested else [],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        }
    return create_initial_state(**initial_state_overrides)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _normalize_format_list(value: Any) -> List[str]:
    normalized: List[str] = []
    for item in _safe_list(value):
        token = str(item).strip().lower()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _is_warning_export_log_entry(entry: Mapping[str, Any]) -> bool:
    code = str(entry.get("code", "")).strip().lower()
    if code.endswith("_warning") or code == "warning":
        return True
    message = str(entry.get("message", "")).strip().lower()
    return "warning" in message and "failed" not in message


def _derive_failed_export_formats(export_metadata: Mapping[str, Any]) -> List[str]:
    explicit_failed = _normalize_format_list(
        export_metadata.get("failed_export_formats")
    )
    if explicit_failed:
        return explicit_failed
    export_status = _safe_dict(export_metadata.get("export_status", {}))
    return [
        str(fmt).strip().lower()
        for fmt, status in export_status.items()
        if str(fmt).strip() and str(status).strip().lower() == "failed"
    ]


def _derive_export_error_count(export_metadata: Mapping[str, Any]) -> int:
    explicit_count = _safe_int(export_metadata.get("export_error_count"), default=-1)
    if explicit_count >= 0:
        return explicit_count
    failed_formats = _derive_failed_export_formats(export_metadata)
    if failed_formats:
        return len(failed_formats)
    error_log = _safe_list(export_metadata.get("error_log", []))
    return sum(
        1
        for item in error_log
        if isinstance(item, Mapping) and not _is_warning_export_log_entry(item)
    )


def _is_recoverable_error(error: Any) -> bool:
    if not isinstance(error, Mapping):
        return False
    return bool(error.get("recoverable", False))


def _status_from_node_update(node_name: str, updates: Mapping[str, Any]) -> str:
    if node_name == ERROR_HANDLER_NODE:
        return "failed"
    if not updates:
        return "skipped"

    if node_name == RESEARCH_AGENT_NODE:
        research_data = _safe_dict(updates.get("research_data", {}))
        if bool(research_data.get("degraded", False)):
            return "degraded"
        return "completed"

    if node_name == IMAGE_AGENT_NODE:
        draft_status = (
            str(_safe_dict(updates.get("draft_status", {})).get("image", ""))
            .strip()
            .lower()
        )
        if draft_status == "skipped":
            return "skipped"
        if draft_status == "failed":
            return "degraded"
        errors = _safe_list(updates.get("errors", []))
        if any(_is_recoverable_error(item) for item in errors):
            return "degraded"
        return "completed"

    if node_name == EXPORT_NODE:
        export_metadata = _safe_dict(updates.get("export_metadata", {}))
        formats_requested = _normalize_format_list(
            export_metadata.get("formats_requested", [])
        )
        if not formats_requested:
            return "skipped"
        if _derive_export_error_count(export_metadata) > 0:
            return "degraded"
        return "completed"

    errors = _safe_list(updates.get("errors", []))
    if errors and any(not _is_recoverable_error(item) for item in errors):
        return "failed"
    if errors and any(_is_recoverable_error(item) for item in errors):
        return "degraded"
    return "completed"


def _message_for_status(node_name: str, status: str) -> str:
    if status == "running":
        return f"{node_name} is running."
    if status == "completed":
        return f"{node_name} completed."
    if status == "skipped":
        return f"{node_name} was skipped."
    if status == "degraded":
        return f"{node_name} completed with warnings."
    if status == "failed":
        return f"{node_name} failed."
    return f"{node_name}: {status}."


def _event_metadata(updates: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    workflow_status = str(updates.get("workflow_status", "")).strip()
    if workflow_status:
        metadata["workflow_status"] = workflow_status
    if "retry_requested" in updates:
        metadata["retry_requested"] = bool(updates.get("retry_requested"))
    if "retry_target" in updates:
        retry_target = str(updates.get("retry_target", "")).strip()
        if retry_target:
            metadata["retry_target"] = retry_target
    if "research_data" in updates:
        metadata["research_degraded"] = bool(
            _safe_dict(updates.get("research_data", {})).get("degraded", False)
        )
    if "export_metadata" in updates:
        export_metadata = _safe_dict(updates.get("export_metadata", {}))
        metadata["export_error_count"] = _derive_export_error_count(export_metadata)
        metadata["export_warning_count"] = _safe_int(
            export_metadata.get("export_warning_count"),
            default=0,
        )
        metadata["requested_export_formats"] = _normalize_format_list(
            export_metadata.get("requested_export_formats")
            or export_metadata.get("formats_requested", [])
        )
        metadata["completed_export_formats"] = _normalize_format_list(
            export_metadata.get("completed_export_formats", [])
        )
        metadata["failed_export_formats"] = _derive_failed_export_formats(
            export_metadata
        )
    return metadata


def _ordered_event_dicts(events: Iterable[UIProgressEvent]) -> List[Dict[str, Any]]:
    return [asdict(event) for event in order_progress_events(events)]


def stream_workflow_progress(
    *,
    user_query: str,
    requested_outputs: List[str] | None = None,
    export_requested: bool = False,
    export_formats: List[str] | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream deterministic workflow progress events and emit a final result.

    Yielded items:
    - {"type": "progress", "event": <UIProgressEvent-as-dict>}
    - {"type": "final", "result": <workflow-state-dict>, "events": [...]}
    """
    initial_state = _build_initial_state(
        user_query=user_query,
        requested_outputs=requested_outputs,
        export_requested=export_requested,
        export_formats=export_formats,
    )
    graph = _get_graph()

    progress_events: List[UIProgressEvent] = []
    latest_state: Dict[str, Any] = deepcopy(initial_state)

    for stream_item in graph.stream(
        deepcopy(initial_state),
        stream_mode=["tasks", "updates", "values"],
    ):
        if not isinstance(stream_item, tuple) or len(stream_item) != 2:
            continue

        stream_kind, payload = stream_item
        if stream_kind == "tasks":
            if not isinstance(payload, dict):
                continue
            node_name = str(payload.get("name", "")).strip()
            is_task_start = "input" in payload and "triggers" in payload
            if node_name not in AUTHORITATIVE_NODE_SET or not is_task_start:
                continue
            running_event = create_progress_event(
                node_name=node_name,
                status="running",
                message=_message_for_status(node_name, "running"),
            )
            progress_events.append(running_event)
            yield {"type": "progress", "event": asdict(running_event)}
            continue

        if stream_kind == "values":
            if isinstance(payload, dict):
                latest_state = deepcopy(payload)
            continue
        if stream_kind != "updates":
            continue
        if not isinstance(payload, dict):
            continue

        for node_name, raw_updates in payload.items():
            if str(node_name).strip() not in AUTHORITATIVE_NODE_SET:
                continue
            updates = _safe_dict(raw_updates)
            status = _status_from_node_update(node_name, updates)
            completed_event = create_progress_event(
                node_name=node_name,
                status=status,
                message=_message_for_status(node_name, status),
                safe_metadata=_event_metadata(updates),
            )
            progress_events.append(completed_event)
            yield {"type": "progress", "event": asdict(completed_event)}

    if not isinstance(latest_state, dict):
        latest_state = {}

    ordered_events = _ordered_event_dicts(progress_events)
    final_result = deepcopy(latest_state)
    final_result["ui_progress_events"] = deepcopy(ordered_events)
    yield {
        "type": "final",
        "result": final_result,
        "events": ordered_events,
    }


def run_workflow(
    *,
    user_query: str,
    requested_outputs: List[str] | None = None,
    export_requested: bool = False,
    export_formats: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Execute orchestration and return a deep-copied result for UI use.

    Frontend should never mutate orchestration internals directly.
    """
    final_result: Dict[str, Any] = {}
    for item in stream_workflow_progress(
        user_query=user_query,
        requested_outputs=requested_outputs,
        export_requested=export_requested,
        export_formats=export_formats,
    ):
        if item.get("type") != "final":
            continue
        payload = item.get("result")
        if isinstance(payload, dict):
            final_result = deepcopy(payload)
    return final_result
