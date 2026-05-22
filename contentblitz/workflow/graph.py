"""LangGraph workflow definition for the ContentBlitz Phase 1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    TypedDict,
)

from langgraph.graph import END as LANGGRAPH_END
from langgraph.graph import START as LANGGRAPH_START
from langgraph.graph import StateGraph

from contentblitz.agents.blog_writer import blog_writer_node
from contentblitz.agents.clarification import clarification_node
from contentblitz.agents.content_strategist import content_strategist_node
from contentblitz.agents.error_handler import error_handler_node
from contentblitz.agents.export import export_node
from contentblitz.agents.image_agent import image_agent_node
from contentblitz.agents.linkedin_writer import linkedin_writer_node
from contentblitz.agents.output_assembler import output_assembler_node
from contentblitz.agents.quality_validator import quality_validator_node
from contentblitz.agents.query_handler import query_handler_node
from contentblitz.agents.research_agent import research_agent_node
from contentblitz.agents.retry_router import retry_router_node
from contentblitz.core.observability import (
    get_workflow_tracer,
    safe_node_end_metadata,
    safe_node_start_metadata,
    safe_workflow_end_metadata,
    safe_workflow_start_metadata,
)
from contentblitz.workflow.routing import (
    AUTHORITATIVE_NODES,
    BLOG_WRITER_NODE,
    CLARIFICATION_NODE,
    CONTENT_STRATEGIST_NODE,
    ERROR_HANDLER_NODE,
    EXPORT_NODE,
    IMAGE_AGENT_NODE,
    LINKEDIN_WRITER_NODE,
    OUTPUT_ASSEMBLER_NODE,
    QUALITY_VALIDATOR_NODE,
    QUERY_HANDLER_NODE,
    RESEARCH_AGENT_NODE,
    RETRY_ROUTER_NODE,
    route_after_query_handler,
    route_from_content_strategist,
    route_from_output_assembler,
    route_from_quality_validator,
    route_from_research_agent,
    route_from_retry_router,
)

# Human-readable aliases for reporting/tests.
START = "START"
END = "END"
WORKFLOW_NODES: List[str] = list(AUTHORITATIVE_NODES)
_UI_STATUS_PRECEDENCE: Dict[str, int] = {
    "pending": 0,
    "running": 1,
    "skipped": 2,
    "completed": 3,
    "degraded": 4,
    "failed": 5,
}


def _safe_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return None


def merge_content_drafts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer for concurrent content_drafts updates from writer fan-out."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if nested_value is None:
                    continue
                nested[nested_key] = nested_value
            merged[key] = nested
        elif isinstance(value, dict):
            merged[key] = dict(value)
        else:
            merged[key] = value
    return merged


def merge_cost_controls(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reducer for concurrent cost_controls updates from writer fan-out.

    This preserves all keys and applies deterministic last-update behavior per key,
    while preventing InvalidUpdateError on parallel writes.
    """
    merged = dict(left or {})
    incoming = dict(right or {})

    monotonic_counter_keys = (
        "tokens_used_this_session",
        "search_queries_used_this_session",
        "image_generations_used_this_session",
        "total_retries_used_this_session",
    )
    cap_keys = (
        "token_budget_per_session",
        "search_query_cap_per_session",
        "image_generation_cap_per_session",
        "max_total_retries_per_session",
    )

    for key in monotonic_counter_keys:
        if key not in merged and key not in incoming:
            continue
        left_value = _safe_non_negative_int(merged.get(key))
        right_value = _safe_non_negative_int(incoming.get(key))
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            merged[key] = right_value
        elif right_value is None:
            merged[key] = left_value
        else:
            merged[key] = max(left_value, right_value)

    for key in cap_keys:
        if key not in merged and key not in incoming:
            continue
        left_value = _safe_non_negative_int(merged.get(key))
        right_value = _safe_non_negative_int(incoming.get(key))
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            merged[key] = right_value
        elif right_value is None:
            merged[key] = left_value
        else:
            # Keep stricter cap on conflicts to avoid accidental limit widening.
            merged[key] = min(left_value, right_value)

    merged["budget_exceeded"] = bool(merged.get("budget_exceeded", False)) or bool(
        incoming.get("budget_exceeded", False)
    )

    handled = set(monotonic_counter_keys) | set(cap_keys) | {"budget_exceeded"}
    for key, value in incoming.items():
        if key in handled:
            continue
        if value is None and key in merged:
            continue
        merged[key] = value
    return merged


def merge_draft_status(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer for concurrent draft_status updates from writer fan-out."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if value is None:
            continue
        merged[key] = value
    return merged


def merge_unique_text_list(left: List[Any], right: List[Any]) -> List[str]:
    """Reducer for user-safe text lists (warnings/status messages)."""
    merged: List[str] = []
    seen: set[str] = set()
    for item in list(left or []) + list(right or []):
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text.lower() in {"none", "null"}:
            continue
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or text.lower() in {"none", "null"}:
        return ""
    return text


def _source_identity(entry: Mapping[str, Any]) -> str:
    url = _clean_text(entry.get("url"))
    if url:
        return f"url:{url.lower()}"
    title = _clean_text(entry.get("title")).lower()
    source = _clean_text(entry.get("source")).lower()
    return f"title_source:{title}|{source}"


def _has_meaningful_source_payload(entry: Mapping[str, Any]) -> bool:
    return any(
        _clean_text(entry.get(field)) for field in ("url", "title", "source", "snippet")
    )


def merge_source_entries(left: List[Any], right: List[Any]) -> List[Dict[str, Any]]:
    """Reducer for source lists with deterministic dedupe and stable ordering."""
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for item in list(left or []) + list(right or []):
        if not isinstance(item, Mapping):
            continue
        candidate = dict(item)
        if not _has_meaningful_source_payload(candidate):
            continue
        identity = _source_identity(candidate)
        # Skip malformed entries that have neither URL nor title+source identity.
        if identity == "title_source:|":
            continue
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(candidate)
    return merged


def _contains_base64_payload(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("data:image/") or "base64," in lowered


def _sanitize_image_output_entry(entry: Mapping[str, Any]) -> Dict[str, Any] | None:
    candidate = dict(entry)
    candidate.pop("base64", None)
    candidate.pop("b64_json", None)

    for key in (
        "url",
        "revised_prompt",
        "prompt",
        "provider",
        "id",
        "status",
        "mime_type",
    ):
        value = candidate.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if key == "url" and cleaned and _contains_base64_payload(cleaned):
                # Drop unsafe embedded payload entries entirely.
                return None
            candidate[key] = cleaned

    # Keep failed entries even without url/id so recoverable failures remain visible.
    if not any(
        candidate.get(field)
        for field in ("status", "provider", "prompt", "id", "url", "error")
    ):
        return None
    return candidate


def _image_output_identity(entry: Mapping[str, Any]) -> str:
    image_id = _clean_text(entry.get("id"))
    if image_id:
        return f"id:{image_id.lower()}"
    url = _clean_text(entry.get("url"))
    if url:
        return f"url:{url.lower()}"
    status = _clean_text(entry.get("status")).lower()
    provider = _clean_text(entry.get("provider")).lower()
    prompt = _clean_text(entry.get("prompt")).lower()
    return f"fallback:{status}|{provider}|{prompt}"


def merge_image_outputs(left: List[Any], right: List[Any]) -> List[Dict[str, Any]]:
    """Reducer for image outputs with deterministic dedupe and base64 stripping."""
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for item in list(left or []) + list(right or []):
        if not isinstance(item, Mapping):
            continue
        sanitized = _sanitize_image_output_entry(item)
        if sanitized is None:
            continue
        identity = _image_output_identity(sanitized)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(sanitized)
    return merged


def merge_nested_dict_skip_none(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge nested dicts without wiping existing values via None updates."""
    merged = dict(left or {})
    for key, value in dict(right or {}).items():
        if value is None:
            continue
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_nested_dict_skip_none(dict(existing), dict(value))
            continue
        merged[key] = value
    return merged


def merge_retry_counts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, int]:
    """Reducer for retry count dictionaries using per-key max semantics."""
    merged: Dict[str, int] = {}
    all_keys = set(dict(left or {}).keys()) | set(dict(right or {}).keys())
    for key in all_keys:
        left_value = _safe_non_negative_int(dict(left or {}).get(key))
        right_value = _safe_non_negative_int(dict(right or {}).get(key))
        if left_value is None and right_value is None:
            continue
        if left_value is None:
            merged[str(key)] = right_value or 0
        elif right_value is None:
            merged[str(key)] = left_value
        else:
            merged[str(key)] = max(left_value, right_value)
    return merged


def _sanitize_progress_metadata(metadata: Any) -> Dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    sanitized: Dict[str, Any] = {}
    for key, value in metadata.items():
        meta_key = _clean_text(str(key))
        if not meta_key:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            sanitized[meta_key] = value
            continue
        if isinstance(value, (str, int, float)):
            sanitized[meta_key] = value
    return sanitized


def merge_progress_events(left: List[Any], right: List[Any]) -> List[Dict[str, Any]]:
    """Reducer for progress events preserving stable order and deduping duplicates."""
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for item in list(left or []) + list(right or []):
        if not isinstance(item, Mapping):
            continue
        node_name = _clean_text(item.get("node_name"))
        status = _clean_text(item.get("status")).lower()
        message = _clean_text(item.get("message"))
        timestamp = _clean_text(item.get("timestamp"))
        if not node_name or not status:
            continue
        dedupe_key = (timestamp, node_name, status, message)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        merged.append(
            {
                "node_name": node_name,
                "status": status,
                "message": message,
                "timestamp": timestamp,
                "safe_metadata": _sanitize_progress_metadata(item.get("safe_metadata")),
            }
        )
    return merged


def _normalize_ui_status(value: Any) -> str:
    normalized = _clean_text(value).lower()
    if normalized in _UI_STATUS_PRECEDENCE:
        return normalized
    return ""


def merge_ui_node_statuses(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Dict[str, str]:
    """Reducer for ui node statuses using deterministic precedence."""
    merged: Dict[str, str] = {}
    base = dict(left or {})
    incoming = dict(right or {})

    for key, value in base.items():
        status = _normalize_ui_status(value)
        if not status:
            continue
        merged[str(key)] = status

    for key, value in incoming.items():
        node = _clean_text(str(key))
        if not node:
            continue
        incoming_status = _normalize_ui_status(value)
        if not incoming_status:
            continue
        existing = merged.get(node, "")
        if not existing:
            merged[node] = incoming_status
            continue
        if _UI_STATUS_PRECEDENCE[incoming_status] >= _UI_STATUS_PRECEDENCE[existing]:
            merged[node] = incoming_status
    return merged


def merge_error_entries(left: List[Any], right: List[Any]) -> List[Dict[str, Any]]:
    """Reducer for normalized error lists produced by concurrent branches."""
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, bool]] = set()
    for item in list(left or []) + list(right or []):
        if not isinstance(item, Mapping):
            continue
        normalized = dict(item)
        message = str(normalized.get("message", "")).strip()
        if not message or message.lower() in {"none", "null"}:
            continue
        normalized["message"] = message
        source = (
            str(normalized.get("agent", normalized.get("node", "unknown")))
            .strip()
            .lower()
        )
        error_type = str(normalized.get("type", "")).strip().lower()
        code = str(normalized.get("code", "")).strip().lower()
        recoverable = bool(normalized.get("recoverable", False))
        dedupe_key = (
            source,
            error_type or code,
            message,
            recoverable,
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        merged.append(normalized)
    return merged


def merge_export_error_log(left: List[Any], right: List[Any]) -> List[Dict[str, Any]]:
    """Reducer for export_metadata.error_log entries."""
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in list(left or []) + list(right or []):
        if not isinstance(item, Mapping):
            continue
        normalized = dict(item)
        fmt = str(normalized.get("format", "")).strip().lower()
        code = str(normalized.get("code", "")).strip().lower() or "export_error"
        message = str(normalized.get("message", "")).strip()
        if not message:
            continue
        normalized["format"] = fmt
        normalized["code"] = code
        normalized["message"] = message
        key = (fmt, code, message)
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def merge_export_metadata(
    left: Dict[str, Any], right: Dict[str, Any]
) -> Dict[str, Any]:
    """Reducer for export metadata that preserves completed format results."""
    merged = dict(left or {})
    incoming = dict(right or {})
    if not incoming:
        return merged

    left_formats = (
        list(merged.get("formats_requested", []))
        if isinstance(merged.get("formats_requested"), list)
        else []
    )
    right_formats = (
        list(incoming.get("formats_requested", []))
        if isinstance(incoming.get("formats_requested"), list)
        else []
    )
    merged["formats_requested"] = list(dict.fromkeys([*left_formats, *right_formats]))

    left_paths = (
        dict(merged.get("export_paths", {}))
        if isinstance(merged.get("export_paths"), Mapping)
        else {}
    )
    right_paths = (
        dict(incoming.get("export_paths", {}))
        if isinstance(incoming.get("export_paths"), Mapping)
        else {}
    )
    left_paths.update({k: v for k, v in right_paths.items() if v})
    merged["export_paths"] = left_paths

    left_status = (
        dict(merged.get("export_status", {}))
        if isinstance(merged.get("export_status"), Mapping)
        else {}
    )
    right_status = (
        dict(incoming.get("export_status", {}))
        if isinstance(incoming.get("export_status"), Mapping)
        else {}
    )
    for fmt in set(left_status) | set(right_status):
        lval = str(left_status.get(fmt, "")).strip().lower()
        rval = str(right_status.get(fmt, "")).strip().lower()
        if "completed" in {lval, rval}:
            left_status[fmt] = "completed"
        elif "failed" in {lval, rval}:
            left_status[fmt] = "failed"
        elif rval:
            left_status[fmt] = rval
        elif lval:
            left_status[fmt] = lval
    merged["export_status"] = left_status

    left_errors = (
        list(merged.get("error_log", []))
        if isinstance(merged.get("error_log"), list)
        else []
    )
    right_errors = (
        list(incoming.get("error_log", []))
        if isinstance(incoming.get("error_log"), list)
        else []
    )
    merged["error_log"] = merge_export_error_log(left_errors, right_errors)

    left_messages = (
        list(merged.get("status_messages", []))
        if isinstance(merged.get("status_messages"), list)
        else []
    )
    right_messages = (
        list(incoming.get("status_messages", []))
        if isinstance(incoming.get("status_messages"), list)
        else []
    )
    merged["status_messages"] = merge_unique_text_list(left_messages, right_messages)

    for key, value in incoming.items():
        if key in {
            "formats_requested",
            "export_paths",
            "export_status",
            "error_log",
            "status_messages",
        }:
            continue
        if value is None and key in merged:
            continue
        merged[key] = value
    return merged


class WorkflowState(TypedDict, total=False):
    session_id: str
    user_id: str
    user_query: str
    intent: str
    routing_decision: str
    requested_outputs: list[str]
    conversation_history: list[Any]
    research_required: bool
    clarification_needed: bool
    clarification_message: Any
    research_data: dict[str, Any]
    sources: Annotated[list[dict[str, Any]], merge_source_entries]
    content_brief: dict[str, dict[str, Any]]
    content_drafts: Annotated[dict[str, dict[str, Any]], merge_content_drafts]
    draft_status: Annotated[dict[str, str], merge_draft_status]
    best_drafts: dict[str, Any]
    attempt_history: dict[str, list[dict[str, Any]]]
    retry_feedback: dict[str, list[str]]
    retry_counts: Annotated[dict[str, int], merge_retry_counts]
    quality_scores: Annotated[dict[str, Any], merge_nested_dict_skip_none]
    image_prompts: Annotated[list[str], merge_unique_text_list]
    image_outputs: Annotated[list[dict[str, Any]], merge_image_outputs]
    tool_outputs: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], merge_error_entries]
    final_response: str
    assembled_outputs: dict[str, Any]
    export_outputs: dict[str, Any]
    workflow_status: str
    export_requested: bool
    export_metadata: Annotated[dict[str, Any], merge_export_metadata]
    cache_metadata: dict[str, Any]
    cost_controls: Annotated[dict[str, Any], merge_cost_controls]
    retry_requested: bool
    retry_target: str
    status_messages: Annotated[list[str], merge_unique_text_list]
    warnings: Annotated[list[str], merge_unique_text_list]
    progress_events: Annotated[list[dict[str, Any]], merge_progress_events]
    ui_node_statuses: Annotated[dict[str, str], merge_ui_node_statuses]
    prompt_injection_detected: bool
    prompt_injection_signals: list[str]
    sanitized_user_query: str


NODE_FUNCTIONS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    QUERY_HANDLER_NODE: query_handler_node,
    CLARIFICATION_NODE: clarification_node,
    RESEARCH_AGENT_NODE: research_agent_node,
    CONTENT_STRATEGIST_NODE: content_strategist_node,
    BLOG_WRITER_NODE: blog_writer_node,
    LINKEDIN_WRITER_NODE: linkedin_writer_node,
    IMAGE_AGENT_NODE: image_agent_node,
    QUALITY_VALIDATOR_NODE: quality_validator_node,
    RETRY_ROUTER_NODE: retry_router_node,
    OUTPUT_ASSEMBLER_NODE: output_assembler_node,
    EXPORT_NODE: export_node,
    ERROR_HANDLER_NODE: error_handler_node,
}


def _merge_state_updates(
    node_name: str,
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Adapt node partial updates into full-state returns for StateGraph(dict).

    This preserves existing state while still allowing nodes to return only
    updates, matching Phase 1 scaffold expectations.
    """

    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        node_started_at = datetime.now(UTC)
        started_at_perf = perf_counter()
        tracer = get_workflow_tracer()
        node_span = tracer.start_node(
            node_name=node_name,
            metadata=safe_node_start_metadata(
                state=state,
                node_name=node_name,
                node_started_at=node_started_at,
            ),
        )
        try:
            updates = node_fn(state)
        except Exception as error:
            node_ended_at = datetime.now(UTC)
            duration_ms = max(0, int((perf_counter() - started_at_perf) * 1000))
            node_span.finish(
                metadata=safe_node_end_metadata(
                    state=state,
                    node_name=node_name,
                    node_status="failed",
                    updates={},
                    node_started_at=node_started_at,
                    node_ended_at=node_ended_at,
                    duration_ms=duration_ms,
                ),
                error=error,
            )
            raise
        if not isinstance(updates, dict):
            node_ended_at = datetime.now(UTC)
            duration_ms = max(0, int((perf_counter() - started_at_perf) * 1000))
            node_span.finish(
                metadata=safe_node_end_metadata(
                    state=state,
                    node_name=node_name,
                    node_status="completed",
                    updates={},
                    node_started_at=node_started_at,
                    node_ended_at=node_ended_at,
                    duration_ms=duration_ms,
                ),
                outputs={"update_keys": []},
            )
            return {}
        partial_updates: Dict[str, Any] = {}
        for key, value in updates.items():
            if state.get(key) != value:
                partial_updates[key] = value
        workflow_status = _clean_text(partial_updates.get("workflow_status"))
        node_status = "completed"
        if node_name == ERROR_HANDLER_NODE:
            node_status = "failed"
        elif workflow_status in {
            "partial_success",
            "degraded",
            "completed_with_warnings",
        }:
            node_status = "degraded"
        node_ended_at = datetime.now(UTC)
        duration_ms = max(0, int((perf_counter() - started_at_perf) * 1000))
        node_span.finish(
            metadata=safe_node_end_metadata(
                state=state,
                node_name=node_name,
                node_status=node_status,
                updates=partial_updates,
                node_started_at=node_started_at,
                node_ended_at=node_ended_at,
                duration_ms=duration_ms,
            ),
            outputs={"update_keys": sorted(partial_updates.keys())},
        )
        return partial_updates

    return _wrapped


# Static graph metadata (authoritative shape).
GRAPH_STRUCTURE: Dict[str, List[str]] = {
    START: [QUERY_HANDLER_NODE],
    QUERY_HANDLER_NODE: [
        CLARIFICATION_NODE,
        IMAGE_AGENT_NODE,
        RESEARCH_AGENT_NODE,
        CONTENT_STRATEGIST_NODE,
        ERROR_HANDLER_NODE,
    ],
    CLARIFICATION_NODE: [END],
    RESEARCH_AGENT_NODE: [
        CONTENT_STRATEGIST_NODE,
        OUTPUT_ASSEMBLER_NODE,
        ERROR_HANDLER_NODE,
    ],
    CONTENT_STRATEGIST_NODE: [
        BLOG_WRITER_NODE,
        LINKEDIN_WRITER_NODE,
        IMAGE_AGENT_NODE,
        ERROR_HANDLER_NODE,
    ],
    BLOG_WRITER_NODE: [QUALITY_VALIDATOR_NODE],
    LINKEDIN_WRITER_NODE: [QUALITY_VALIDATOR_NODE],
    IMAGE_AGENT_NODE: [QUALITY_VALIDATOR_NODE],
    QUALITY_VALIDATOR_NODE: [
        RETRY_ROUTER_NODE,
        OUTPUT_ASSEMBLER_NODE,
        ERROR_HANDLER_NODE,
    ],
    RETRY_ROUTER_NODE: [
        BLOG_WRITER_NODE,
        LINKEDIN_WRITER_NODE,
        IMAGE_AGENT_NODE,
        OUTPUT_ASSEMBLER_NODE,
    ],
    OUTPUT_ASSEMBLER_NODE: [EXPORT_NODE, ERROR_HANDLER_NODE],
    EXPORT_NODE: [END],
    ERROR_HANDLER_NODE: [END],
}

ROUTING_TABLE: Dict[str, str] = {
    QUERY_HANDLER_NODE: "route_after_query_handler",
    RESEARCH_AGENT_NODE: "route_from_research_agent",
    CONTENT_STRATEGIST_NODE: "route_from_content_strategist",
    QUALITY_VALIDATOR_NODE: "route_from_quality_validator",
    RETRY_ROUTER_NODE: "route_from_retry_router",
    OUTPUT_ASSEMBLER_NODE: "_route_from_output_assembler_for_graph",
}


@dataclass(frozen=True)
class WorkflowGraph:
    nodes: List[str]
    edges: Dict[str, List[str]]
    routing_table: Dict[str, str]


class _TracedCompiledGraph:
    """Proxy around compiled LangGraph with optional workflow-level tracing."""

    def __init__(self, compiled_graph: Any) -> None:
        self._compiled_graph = compiled_graph

    def __getattr__(self, name: str) -> Any:
        return getattr(self._compiled_graph, name)

    def invoke(self, state: Any, *args: Any, **kwargs: Any) -> Any:
        initial_state = state if isinstance(state, Mapping) else {}
        tracer = get_workflow_tracer()
        workflow_span = tracer.start_workflow(
            metadata=safe_workflow_start_metadata(initial_state),
        )
        try:
            result = self._compiled_graph.invoke(state, *args, **kwargs)
        except Exception as error:
            workflow_span.finish(
                metadata=safe_workflow_end_metadata(
                    initial_state=initial_state,
                    final_state=None,
                ),
                error=error,
            )
            raise

        final_state = result if isinstance(result, Mapping) else {}
        end_metadata = safe_workflow_end_metadata(
            initial_state=initial_state,
            final_state=final_state,
        )
        workflow_status = _clean_text(final_state.get("workflow_status"))
        workflow_span.finish(
            metadata=end_metadata,
            outputs={"workflow_status": workflow_status},
        )
        return result

    def stream(self, state: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        initial_state = state if isinstance(state, Mapping) else {}
        tracer = get_workflow_tracer()
        workflow_span = tracer.start_workflow(
            metadata=safe_workflow_start_metadata(initial_state),
        )
        latest_state: Mapping[str, Any] = initial_state
        stream_iterator = self._compiled_graph.stream(state, *args, **kwargs)
        stream_error: BaseException | None = None
        try:
            for item in stream_iterator:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and item[0] == "values"
                    and isinstance(item[1], Mapping)
                ):
                    latest_state = dict(item[1])
                yield item
        except Exception as error:
            stream_error = error
            raise
        finally:
            workflow_status = (
                _clean_text(latest_state.get("workflow_status"))
                if isinstance(latest_state, Mapping)
                else ""
            )
            if stream_error is not None:
                workflow_span.finish(
                    metadata=safe_workflow_end_metadata(
                        initial_state=initial_state,
                        final_state=latest_state,
                    ),
                    error=stream_error,
                )
            else:
                workflow_span.finish(
                    metadata=safe_workflow_end_metadata(
                        initial_state=initial_state,
                        final_state=latest_state,
                    ),
                    outputs={"workflow_status": workflow_status},
                )

    async def ainvoke(self, state: Any, *args: Any, **kwargs: Any) -> Any:
        initial_state = state if isinstance(state, Mapping) else {}
        tracer = get_workflow_tracer()
        workflow_span = tracer.start_workflow(
            metadata=safe_workflow_start_metadata(initial_state),
        )
        try:
            result = await self._compiled_graph.ainvoke(state, *args, **kwargs)
        except Exception as error:
            workflow_span.finish(
                metadata=safe_workflow_end_metadata(
                    initial_state=initial_state,
                    final_state=None,
                ),
                error=error,
            )
            raise

        final_state = result if isinstance(result, Mapping) else {}
        end_metadata = safe_workflow_end_metadata(
            initial_state=initial_state,
            final_state=final_state,
        )
        workflow_status = _clean_text(final_state.get("workflow_status"))
        workflow_span.finish(
            metadata=end_metadata,
            outputs={"workflow_status": workflow_status},
        )
        return result

    async def astream(
        self,
        state: Any,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        initial_state = state if isinstance(state, Mapping) else {}
        tracer = get_workflow_tracer()
        workflow_span = tracer.start_workflow(
            metadata=safe_workflow_start_metadata(initial_state),
        )
        latest_state: Mapping[str, Any] = initial_state
        stream_error: BaseException | None = None

        try:
            async for item in self._compiled_graph.astream(state, *args, **kwargs):
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and item[0] == "values"
                    and isinstance(item[1], Mapping)
                ):
                    latest_state = dict(item[1])
                yield item
        except Exception as error:
            stream_error = error
            raise
        finally:
            workflow_status = (
                _clean_text(latest_state.get("workflow_status"))
                if isinstance(latest_state, Mapping)
                else ""
            )
            if stream_error is not None:
                workflow_span.finish(
                    metadata=safe_workflow_end_metadata(
                        initial_state=initial_state,
                        final_state=latest_state,
                    ),
                    error=stream_error,
                )
            else:
                workflow_span.finish(
                    metadata=safe_workflow_end_metadata(
                        initial_state=initial_state,
                        final_state=latest_state,
                    ),
                    outputs={"workflow_status": workflow_status},
                )


def _route_from_output_assembler_for_graph(state: Mapping[str, Any]) -> str:
    """
    Convert routing decision to graph edge target.
    - error state -> error_handler_node
    - otherwise -> export_node
      (export node is responsible for no-op behavior when nothing is exportable)
    """
    decision = route_from_output_assembler(state)
    if decision == ERROR_HANDLER_NODE:
        return ERROR_HANDLER_NODE
    return EXPORT_NODE


def build_workflow_graph() -> WorkflowGraph:
    """Return static graph metadata used for tests and validation."""
    return WorkflowGraph(
        nodes=list(WORKFLOW_NODES),
        edges={key: list(value) for key, value in GRAPH_STRUCTURE.items()},
        routing_table=dict(ROUTING_TABLE),
    )


def build_langgraph() -> Any:
    """Build and compile the executable LangGraph workflow."""
    graph = StateGraph(WorkflowState)

    for node_name, node_fn in NODE_FUNCTIONS.items():
        graph.add_node(node_name, _merge_state_updates(node_name, node_fn))

    graph.add_edge(LANGGRAPH_START, QUERY_HANDLER_NODE)

    graph.add_conditional_edges(QUERY_HANDLER_NODE, route_after_query_handler)
    graph.add_edge(CLARIFICATION_NODE, LANGGRAPH_END)

    graph.add_conditional_edges(RESEARCH_AGENT_NODE, route_from_research_agent)
    graph.add_conditional_edges(CONTENT_STRATEGIST_NODE, route_from_content_strategist)

    graph.add_edge(BLOG_WRITER_NODE, QUALITY_VALIDATOR_NODE)
    graph.add_edge(LINKEDIN_WRITER_NODE, QUALITY_VALIDATOR_NODE)
    graph.add_edge(IMAGE_AGENT_NODE, QUALITY_VALIDATOR_NODE)

    graph.add_conditional_edges(QUALITY_VALIDATOR_NODE, route_from_quality_validator)
    graph.add_conditional_edges(RETRY_ROUTER_NODE, route_from_retry_router)

    graph.add_conditional_edges(
        OUTPUT_ASSEMBLER_NODE, _route_from_output_assembler_for_graph
    )
    graph.add_edge(EXPORT_NODE, LANGGRAPH_END)
    graph.add_edge(ERROR_HANDLER_NODE, LANGGRAPH_END)

    compiled = graph.compile()
    return _TracedCompiledGraph(compiled)
