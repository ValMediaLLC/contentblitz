"""LangGraph workflow definition for the ContentBlitz Phase 1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Callable, Dict, List, Mapping, TypedDict

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
from contentblitz.agents.query_handler import query_handler_node
from contentblitz.agents.quality_validator import quality_validator_node
from contentblitz.agents.research_agent import research_agent_node
from contentblitz.agents.retry_router import retry_router_node
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
    QUERY_HANDLER_NODE,
    QUALITY_VALIDATOR_NODE,
    RESEARCH_AGENT_NODE,
    RETRY_ROUTER_NODE,
    route_from_content_strategist,
    route_from_output_assembler,
    route_from_quality_validator,
    route_after_query_handler,
    route_from_research_agent,
    route_from_retry_router,
)

# Human-readable aliases for reporting/tests.
START = "START"
END = "END"
WORKFLOW_NODES: List[str] = list(AUTHORITATIVE_NODES)


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
        text = str(item).strip()
        if not text or text.lower() in {"none", "null"}:
            continue
        if text in seen:
            continue
        seen.add(text)
        merged.append(text)
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
        source = str(
            normalized.get("agent", normalized.get("node", "unknown"))
        ).strip().lower()
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


def merge_export_metadata(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer for export metadata that preserves completed format results."""
    merged = dict(left or {})
    incoming = dict(right or {})
    if not incoming:
        return merged

    left_formats = list(merged.get("formats_requested", [])) if isinstance(merged.get("formats_requested"), list) else []
    right_formats = list(incoming.get("formats_requested", [])) if isinstance(incoming.get("formats_requested"), list) else []
    merged["formats_requested"] = list(dict.fromkeys([*left_formats, *right_formats]))

    left_paths = dict(merged.get("export_paths", {})) if isinstance(merged.get("export_paths"), Mapping) else {}
    right_paths = dict(incoming.get("export_paths", {})) if isinstance(incoming.get("export_paths"), Mapping) else {}
    left_paths.update({k: v for k, v in right_paths.items() if v})
    merged["export_paths"] = left_paths

    left_status = dict(merged.get("export_status", {})) if isinstance(merged.get("export_status"), Mapping) else {}
    right_status = dict(incoming.get("export_status", {})) if isinstance(incoming.get("export_status"), Mapping) else {}
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

    left_errors = list(merged.get("error_log", [])) if isinstance(merged.get("error_log"), list) else []
    right_errors = list(incoming.get("error_log", [])) if isinstance(incoming.get("error_log"), list) else []
    merged["error_log"] = merge_export_error_log(left_errors, right_errors)

    left_messages = list(merged.get("status_messages", [])) if isinstance(merged.get("status_messages"), list) else []
    right_messages = list(incoming.get("status_messages", [])) if isinstance(incoming.get("status_messages"), list) else []
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
    sources: list[dict[str, Any]]
    content_brief: dict[str, dict[str, Any]]
    content_drafts: Annotated[dict[str, dict[str, Any]], merge_content_drafts]
    draft_status: Annotated[dict[str, str], merge_draft_status]
    best_drafts: dict[str, Any]
    attempt_history: dict[str, list[dict[str, Any]]]
    retry_feedback: dict[str, list[str]]
    retry_counts: dict[str, int]
    quality_scores: dict[str, Any]
    image_prompts: list[str]
    image_outputs: list[dict[str, Any]]
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
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Adapt node partial updates into full-state returns for StateGraph(dict).

    This preserves existing state while still allowing nodes to return only
    updates, matching Phase 1 scaffold expectations.
    """

    def _wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        updates = node_fn(state)
        if not isinstance(updates, dict):
            return {}
        partial_updates: Dict[str, Any] = {}
        for key, value in updates.items():
            if state.get(key) != value:
                partial_updates[key] = value
        return partial_updates

    return _wrapped

# Static graph metadata (authoritative shape).
GRAPH_STRUCTURE: Dict[str, List[str]] = {
    START: [QUERY_HANDLER_NODE],
    QUERY_HANDLER_NODE: [CLARIFICATION_NODE, IMAGE_AGENT_NODE, RESEARCH_AGENT_NODE, CONTENT_STRATEGIST_NODE, ERROR_HANDLER_NODE],
    CLARIFICATION_NODE: [END],
    RESEARCH_AGENT_NODE: [CONTENT_STRATEGIST_NODE, OUTPUT_ASSEMBLER_NODE, ERROR_HANDLER_NODE],
    CONTENT_STRATEGIST_NODE: [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE, IMAGE_AGENT_NODE, ERROR_HANDLER_NODE],
    BLOG_WRITER_NODE: [QUALITY_VALIDATOR_NODE],
    LINKEDIN_WRITER_NODE: [QUALITY_VALIDATOR_NODE],
    IMAGE_AGENT_NODE: [QUALITY_VALIDATOR_NODE],
    QUALITY_VALIDATOR_NODE: [RETRY_ROUTER_NODE, OUTPUT_ASSEMBLER_NODE, ERROR_HANDLER_NODE],
    RETRY_ROUTER_NODE: [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE, IMAGE_AGENT_NODE, OUTPUT_ASSEMBLER_NODE],
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
        graph.add_node(node_name, _merge_state_updates(node_fn))

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

    graph.add_conditional_edges(OUTPUT_ASSEMBLER_NODE, _route_from_output_assembler_for_graph)
    graph.add_edge(EXPORT_NODE, LANGGRAPH_END)
    graph.add_edge(ERROR_HANDLER_NODE, LANGGRAPH_END)

    return graph.compile()
