"""Deterministic workflow routing functions for ContentBlitz."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Tuple, Union

from contentblitz.core.router import route_with_retry

QUERY_HANDLER_NODE = "query_handler_node"
CLARIFICATION_NODE = "clarification_node"
RESEARCH_AGENT_NODE = "research_agent_node"
CONTENT_STRATEGIST_NODE = "content_strategist_node"
BLOG_WRITER_NODE = "blog_writer_node"
LINKEDIN_WRITER_NODE = "linkedin_writer_node"
IMAGE_AGENT_NODE = "image_agent_node"
QUALITY_VALIDATOR_NODE = "quality_validator_node"
RETRY_ROUTER_NODE = "retry_router_node"
OUTPUT_ASSEMBLER_NODE = "output_assembler_node"
EXPORT_NODE = "export_node"
ERROR_HANDLER_NODE = "error_handler_node"

AUTHORITATIVE_NODES: List[str] = [
    QUERY_HANDLER_NODE,
    CLARIFICATION_NODE,
    RESEARCH_AGENT_NODE,
    CONTENT_STRATEGIST_NODE,
    BLOG_WRITER_NODE,
    LINKEDIN_WRITER_NODE,
    IMAGE_AGENT_NODE,
    QUALITY_VALIDATOR_NODE,
    RETRY_ROUTER_NODE,
    OUTPUT_ASSEMBLER_NODE,
    EXPORT_NODE,
    ERROR_HANDLER_NODE,
]
AUTHORITATIVE_NODE_SET = set(AUTHORITATIVE_NODES)

WRITER_NODES = [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE, IMAGE_AGENT_NODE]
WRITER_NODE_SET = set(WRITER_NODES)

RouteDecision = Union[str, List[str]]


def _requested_outputs(state: Mapping[str, Any]) -> set[str]:
    raw_outputs = state.get("requested_outputs", [])
    if not isinstance(raw_outputs, list):
        return set()
    return {str(item).strip().lower() for item in raw_outputs if str(item).strip()}


def _has_errors(state: Mapping[str, Any]) -> bool:
    errors = state.get("errors", [])
    return isinstance(errors, list) and len(errors) > 0


def is_authoritative_node(node: str) -> bool:
    return node in AUTHORITATIVE_NODE_SET


def should_clarify(state: Mapping[str, Any]) -> bool:
    return bool(state.get("clarification_needed", False))


def should_research(state: Mapping[str, Any]) -> bool:
    outputs = _requested_outputs(state)
    return bool(state.get("research_required", False)) or "research" in outputs


def should_generate_image(state: Mapping[str, Any]) -> bool:
    outputs = _requested_outputs(state)
    return "image" in outputs


def should_retry(state: Mapping[str, Any]) -> bool:
    return bool(state.get("retry_requested", False))


def should_export(state: Mapping[str, Any]) -> bool:
    return bool(state.get("export_requested", False))


def is_image_only_request(state: Mapping[str, Any]) -> bool:
    return _requested_outputs(state) == {"image"}


def is_research_only_request(state: Mapping[str, Any]) -> bool:
    return _requested_outputs(state) == {"research"}


def route_from_query_handler(state: Mapping[str, Any]) -> str:
    """Primary router: clarification -> image-only -> research -> strategist."""
    if _has_errors(state):
        return ERROR_HANDLER_NODE
    if should_clarify(state):
        return CLARIFICATION_NODE
    if is_image_only_request(state):
        return IMAGE_AGENT_NODE
    if should_research(state):
        return RESEARCH_AGENT_NODE
    return CONTENT_STRATEGIST_NODE


def route_from_research_agent(state: Mapping[str, Any]) -> str:
    """Research-only requests bypass strategists/writers and go straight to assembly."""
    if _has_errors(state):
        return ERROR_HANDLER_NODE
    if is_research_only_request(state):
        return OUTPUT_ASSEMBLER_NODE
    return CONTENT_STRATEGIST_NODE


def route_from_content_strategist(state: Mapping[str, Any]) -> List[str]:
    """Fan out to requested writer nodes. Defaults to blog+linkedin."""
    if _has_errors(state):
        return [ERROR_HANDLER_NODE]

    outputs = _requested_outputs(state)
    routes: List[str] = []

    if not outputs or "blog" in outputs:
        routes.append(BLOG_WRITER_NODE)
    if not outputs or "linkedin" in outputs:
        routes.append(LINKEDIN_WRITER_NODE)
    if "image" in outputs:
        routes.append(IMAGE_AGENT_NODE)

    if not routes:
        routes = [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE]

    return routes


def route_from_quality_validator(state: Mapping[str, Any]) -> str:
    if _has_errors(state):
        return ERROR_HANDLER_NODE
    if should_retry(state):
        return RETRY_ROUTER_NODE
    return OUTPUT_ASSEMBLER_NODE


def _retry_target(state: Mapping[str, Any]) -> Tuple[str, str]:
    target = str(state.get("retry_target", "")).strip().lower()
    target_map: Dict[str, Tuple[str, str]] = {
        "blog": (BLOG_WRITER_NODE, "blog_writer"),
        BLOG_WRITER_NODE: (BLOG_WRITER_NODE, "blog_writer"),
        "linkedin": (LINKEDIN_WRITER_NODE, "linkedin_writer"),
        LINKEDIN_WRITER_NODE: (LINKEDIN_WRITER_NODE, "linkedin_writer"),
        "image": (IMAGE_AGENT_NODE, "image_agent"),
        IMAGE_AGENT_NODE: (IMAGE_AGENT_NODE, "image_agent"),
    }
    return target_map.get(target, (OUTPUT_ASSEMBLER_NODE, "output_assembler"))


def route_from_retry_router(state: MutableMapping[str, Any]) -> str:
    """
    Retry router with ordering rule:
    retry counter increments before route decision.
    """
    target_node, agent_key = _retry_target(state)
    if target_node == OUTPUT_ASSEMBLER_NODE:
        return OUTPUT_ASSEMBLER_NODE

    route = route_with_retry(
        state=state,
        agent_key=agent_key,
        retry_node=target_node,
        exhausted_node=OUTPUT_ASSEMBLER_NODE,
    )
    if route in WRITER_NODE_SET or route == OUTPUT_ASSEMBLER_NODE:
        return route
    return OUTPUT_ASSEMBLER_NODE


def route_from_output_assembler(state: Mapping[str, Any]) -> str:
    """Optional export path. Non-export flows terminate directly at END in graph."""
    if _has_errors(state):
        return ERROR_HANDLER_NODE
    if should_export(state):
        return EXPORT_NODE
    return OUTPUT_ASSEMBLER_NODE


def validate_route_decision(decision: RouteDecision) -> bool:
    """Validate a route decision only contains authoritative nodes."""
    if isinstance(decision, str):
        return decision in AUTHORITATIVE_NODE_SET
    if isinstance(decision, list):
        if not decision:
            return False
        return all(item in AUTHORITATIVE_NODE_SET for item in decision)
    return False

