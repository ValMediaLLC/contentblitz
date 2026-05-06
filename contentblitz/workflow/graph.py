"""LangGraph workflow definition for the ContentBlitz Phase 1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, TypedDict

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
    content_drafts: dict[str, dict[str, Any]]
    best_drafts: dict[str, Any]
    attempt_history: dict[str, list[dict[str, Any]]]
    retry_feedback: dict[str, list[str]]
    retry_counts: dict[str, int]
    quality_scores: dict[str, Any]
    image_prompts: list[str]
    image_outputs: list[dict[str, Any]]
    tool_outputs: dict[str, Any]
    errors: list[dict[str, Any]]
    final_response: str
    workflow_status: str
    export_requested: bool
    export_metadata: dict[str, Any]
    cache_metadata: dict[str, Any]
    cost_controls: dict[str, Any]
    retry_requested: bool
    retry_target: str

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
    OUTPUT_ASSEMBLER_NODE: [EXPORT_NODE, END, ERROR_HANDLER_NODE],
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
    - export requested -> export_node
    - error state -> error_handler_node
    - otherwise -> END
    """
    decision = route_from_output_assembler(state)
    if decision == EXPORT_NODE:
        return EXPORT_NODE
    if decision == ERROR_HANDLER_NODE:
        return ERROR_HANDLER_NODE
    return LANGGRAPH_END


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
