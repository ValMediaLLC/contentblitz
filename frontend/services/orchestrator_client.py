"""Thin frontend adapter over ContentBlitz orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from time import monotonic
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


def _safe_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return None


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


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


def _node_provider_and_model(
    *,
    node_name: str,
    updates: Mapping[str, Any],
) -> tuple[str, str]:
    provider = ""
    model = ""

    if node_name == "blog_writer_node":
        blog = _safe_dict(_safe_dict(updates.get("content_drafts", {})).get("blog", {}))
        model = _safe_text(blog.get("model_used"))
        if model:
            provider = "openai"
    elif node_name == "linkedin_writer_node":
        linkedin = _safe_dict(
            _safe_dict(updates.get("content_drafts", {})).get("linkedin", {})
        )
        model = _safe_text(linkedin.get("model_used"))
        if model:
            provider = "openai"
    elif node_name == "image_agent_node":
        tool_outputs = _safe_dict(updates.get("tool_outputs", {}))
        image_tool = _safe_dict(tool_outputs.get("image_agent", {}))
        provider = _safe_text(image_tool.get("provider")).lower()
        model = _safe_text(image_tool.get("provider"))
        if not provider:
            image_outputs = _safe_list(updates.get("image_outputs", []))
            if image_outputs and isinstance(image_outputs[0], Mapping):
                provider = _safe_text(image_outputs[0].get("provider")).lower()
                model = _safe_text(image_outputs[0].get("provider"))
    elif node_name == "research_agent_node":
        sources = _safe_list(updates.get("sources", []))
        for item in sources:
            if not isinstance(item, Mapping):
                continue
            provider = _safe_text(item.get("provider") or item.get("source")).lower()
            if provider:
                break
    elif node_name == "query_handler_node":
        query_handler_output = _safe_dict(
            _safe_dict(updates.get("tool_outputs", {})).get("query_handler", {})
        )
        provider = _safe_text(query_handler_output.get("provider")).lower()
        model = _safe_text(query_handler_output.get("model"))
    elif node_name == "content_strategist_node":
        strategist_output = _safe_dict(
            _safe_dict(updates.get("tool_outputs", {})).get("content_strategist", {})
        )
        provider = _safe_text(strategist_output.get("provider")).lower()
        model = _safe_text(strategist_output.get("model"))

    return provider, model


def _explicit_provider_latency_from_updates(
    *,
    node_name: str,
    updates: Mapping[str, Any],
) -> int | None:
    return _explicit_provider_performance_from_updates(
        node_name=node_name,
        updates=updates,
    )[0]


def _explicit_provider_call_count_from_updates(
    *,
    node_name: str,
    updates: Mapping[str, Any],
) -> int | None:
    return _explicit_provider_performance_from_updates(
        node_name=node_name,
        updates=updates,
    )[1]


def _explicit_provider_performance_from_updates(
    *,
    node_name: str,
    updates: Mapping[str, Any],
) -> tuple[int | None, int | None]:
    def _metrics_from_payload(
        payload: Mapping[str, Any],
    ) -> tuple[int | None, int | None]:
        return (
            _safe_non_negative_int(payload.get("provider_latency_ms")),
            _safe_non_negative_int(payload.get("provider_call_count")),
        )

    direct_latency = _safe_non_negative_int(updates.get("provider_latency_ms"))
    direct_count = _safe_non_negative_int(updates.get("provider_call_count"))
    if direct_latency is not None or direct_count is not None:
        return direct_latency, direct_count

    tool_outputs = _safe_dict(updates.get("tool_outputs", {}))
    if node_name == "blog_writer_node":
        blog = _safe_dict(_safe_dict(updates.get("content_drafts", {})).get("blog", {}))
        return _metrics_from_payload(blog)
    if node_name == "linkedin_writer_node":
        linkedin = _safe_dict(
            _safe_dict(updates.get("content_drafts", {})).get("linkedin", {})
        )
        return _metrics_from_payload(linkedin)
    if node_name == "research_agent_node":
        research_data = _safe_dict(updates.get("research_data", {}))
        return (
            None,
            _safe_non_negative_int(research_data.get("provider_call_count")),
        )
    if node_name == "image_agent_node":
        image_output = _safe_dict(tool_outputs.get("image_agent", {}))
        return _metrics_from_payload(image_output)
    if node_name == "query_handler_node":
        query_handler_output = _safe_dict(tool_outputs.get("query_handler", {}))
        return _metrics_from_payload(query_handler_output)
    if node_name == "content_strategist_node":
        strategist_output = _safe_dict(tool_outputs.get("content_strategist", {}))
        return _metrics_from_payload(strategist_output)
    return None, None


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


def _event_metadata(
    updates: Mapping[str, Any],
    *,
    node_name: str,
    status: str,
    node_started_at: str,
    node_ended_at: str,
    duration_ms: int,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    workflow_status = str(updates.get("workflow_status", "")).strip()
    if workflow_status:
        metadata["workflow_status"] = workflow_status
    metadata["node_started_at"] = node_started_at
    metadata["node_ended_at"] = node_ended_at
    metadata["duration_ms"] = max(0, int(duration_ms))
    if "retry_requested" in updates:
        metadata["retry_requested"] = bool(updates.get("retry_requested"))
    if "retry_target" in updates:
        retry_target = str(updates.get("retry_target", "")).strip()
        if retry_target:
            metadata["retry_target"] = retry_target
    if "research_data" in updates:
        research_data = _safe_dict(updates.get("research_data", {}))
        metadata["research_degraded"] = bool(research_data.get("degraded", False))
        if "cache_hit" in research_data:
            metadata["cache_hit"] = bool(research_data.get("cache_hit"))
        provider_latency_total_ms = _safe_non_negative_int(
            research_data.get("provider_latency_total_ms")
        )
        if provider_latency_total_ms is not None:
            metadata["provider_latency_total_ms"] = provider_latency_total_ms
        provider_latency_wall_ms = _safe_non_negative_int(
            research_data.get("provider_latency_wall_ms")
        )
        if provider_latency_wall_ms is not None:
            metadata["provider_latency_wall_ms"] = provider_latency_wall_ms
        latency_by_provider = research_data.get("provider_latency_by_provider_ms")
        if isinstance(latency_by_provider, Mapping):
            safe_latency_by_provider: Dict[str, int] = {}
            for raw_key, raw_value in latency_by_provider.items():
                key = _safe_text(raw_key).lower()
                value = _safe_non_negative_int(raw_value)
                if key and value is not None:
                    safe_latency_by_provider[key] = value
            if safe_latency_by_provider:
                metadata["provider_latency_by_provider_ms"] = safe_latency_by_provider
        call_count_by_provider = research_data.get("provider_call_count_by_provider")
        if isinstance(call_count_by_provider, Mapping):
            safe_call_count_by_provider: Dict[str, int] = {}
            for raw_key, raw_value in call_count_by_provider.items():
                key = _safe_text(raw_key).lower()
                value = _safe_non_negative_int(raw_value)
                if key and value is not None:
                    safe_call_count_by_provider[key] = value
            if safe_call_count_by_provider:
                metadata["provider_call_count_by_provider"] = (
                    safe_call_count_by_provider
                )
        provider_timeout_count = _safe_non_negative_int(
            research_data.get("provider_timeout_count")
        )
        if provider_timeout_count is not None:
            metadata["provider_timeout_count"] = provider_timeout_count
        timeout_count_by_provider = research_data.get(
            "provider_timeout_count_by_provider"
        )
        if isinstance(timeout_count_by_provider, Mapping):
            safe_timeout_count_by_provider: Dict[str, int] = {}
            for raw_key, raw_value in timeout_count_by_provider.items():
                key = _safe_text(raw_key).lower()
                value = _safe_non_negative_int(raw_value)
                if key and value is not None:
                    safe_timeout_count_by_provider[key] = value
            if safe_timeout_count_by_provider:
                metadata["provider_timeout_count_by_provider"] = (
                    safe_timeout_count_by_provider
                )
        search_provider_wall_timeout_ms = _safe_non_negative_int(
            research_data.get("search_provider_wall_timeout_ms")
        )
        if search_provider_wall_timeout_ms is not None:
            metadata["search_provider_wall_timeout_ms"] = (
                search_provider_wall_timeout_ms
            )
        if "search_provider_wall_timeout_triggered" in research_data:
            metadata["search_provider_wall_timeout_triggered"] = bool(
                research_data.get("search_provider_wall_timeout_triggered")
            )
    if node_name == "content_strategist_node":
        strategist_output = _safe_dict(
            _safe_dict(updates.get("tool_outputs", {})).get("content_strategist", {})
        )
        provider_latency_total_ms = _safe_non_negative_int(
            strategist_output.get("provider_latency_total_ms")
        )
        if provider_latency_total_ms is not None:
            metadata["provider_latency_total_ms"] = provider_latency_total_ms
        provider_latency_wall_ms = _safe_non_negative_int(
            strategist_output.get("provider_latency_wall_ms")
        )
        if provider_latency_wall_ms is not None:
            metadata["provider_latency_wall_ms"] = provider_latency_wall_ms

        latency_by_output_type = strategist_output.get(
            "provider_latency_by_output_type_ms"
        )
        if isinstance(latency_by_output_type, Mapping):
            safe_latency_by_output_type: Dict[str, int] = {}
            for raw_key, raw_value in latency_by_output_type.items():
                key = _safe_text(raw_key).lower()
                value = _safe_non_negative_int(raw_value)
                if key and value is not None:
                    safe_latency_by_output_type[key] = value
            if safe_latency_by_output_type:
                metadata["provider_latency_by_output_type_ms"] = (
                    safe_latency_by_output_type
                )

        call_count_by_output_type = strategist_output.get(
            "provider_call_count_by_output_type"
        )
        if isinstance(call_count_by_output_type, Mapping):
            safe_call_count_by_output_type: Dict[str, int] = {}
            for raw_key, raw_value in call_count_by_output_type.items():
                key = _safe_text(raw_key).lower()
                value = _safe_non_negative_int(raw_value)
                if key and value is not None:
                    safe_call_count_by_output_type[key] = value
            if safe_call_count_by_output_type:
                metadata["provider_call_count_by_output_type"] = (
                    safe_call_count_by_output_type
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
    provider, model = _node_provider_and_model(node_name=node_name, updates=updates)
    if provider:
        metadata["provider"] = provider
    if model:
        metadata["model"] = model
    provider_latency_ms = _explicit_provider_latency_from_updates(
        node_name=node_name,
        updates=updates,
    )
    if provider_latency_ms is not None:
        metadata["provider_latency_ms"] = provider_latency_ms
    provider_call_count = _explicit_provider_call_count_from_updates(
        node_name=node_name,
        updates=updates,
    )
    if provider_call_count is not None:
        metadata["provider_call_count"] = provider_call_count
    metadata["node_status"] = str(status).strip().lower()
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
    node_start_times: Dict[str, tuple[float, str]] = {}

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
                safe_metadata={
                    "node_started_at": datetime.now(UTC).isoformat(
                        timespec="milliseconds"
                    )
                },
            )
            node_start_times[node_name] = (monotonic(), running_event.timestamp)
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
            started_perf, started_at_iso = node_start_times.pop(
                node_name,
                (monotonic(), ""),
            )
            ended_at_iso = datetime.now(UTC).isoformat(timespec="milliseconds")
            duration_ms = max(0, int((monotonic() - started_perf) * 1000))
            if not started_at_iso:
                started_at_iso = ended_at_iso
            completed_event = create_progress_event(
                node_name=node_name,
                status=status,
                message=_message_for_status(node_name, status),
                timestamp=ended_at_iso,
                safe_metadata=_event_metadata(
                    updates,
                    node_name=node_name,
                    status=status,
                    node_started_at=started_at_iso,
                    node_ended_at=ended_at_iso,
                    duration_ms=duration_ms,
                ),
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
