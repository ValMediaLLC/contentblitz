import pytest

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import (
    END,
    GRAPH_STRUCTURE,
    OUTPUT_ASSEMBLER_NODE,
    RETRY_ROUTER_NODE,
    START,
    WORKFLOW_NODES,
    build_langgraph,
    build_workflow_graph,
)
from contentblitz.workflow.routing import (
    BLOG_WRITER_NODE,
    CLARIFICATION_NODE,
    IMAGE_AGENT_NODE,
    LINKEDIN_WRITER_NODE,
    OUTPUT_ASSEMBLER_NODE as OUTPUT_ASSEMBLER_ROUTE_NODE,
    RESEARCH_AGENT_NODE,
    route_from_query_handler,
    route_from_research_agent,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:The default value of `allowed_objects` will change"
)


def _executed_nodes(compiled_graph, state) -> list[str]:
    events = list(compiled_graph.stream(state, stream_mode="updates"))
    executed: list[str] = []
    for event in events:
        executed.extend(list(event.keys()))
    return executed


def test_graph_metadata_includes_12_authoritative_nodes_and_start_end() -> None:
    graph = build_workflow_graph()
    assert len(graph.nodes) == 12
    assert set(graph.nodes) == set(WORKFLOW_NODES)
    assert START in graph.edges
    assert END in graph.edges[CLARIFICATION_NODE]


def test_image_only_routes_query_handler_to_image_agent() -> None:
    state = create_initial_state(requested_outputs=["image"])
    assert route_from_query_handler(state) == IMAGE_AGENT_NODE

    compiled = build_langgraph()
    executed = _executed_nodes(compiled, state)
    assert executed[0] == "query_handler_node"
    assert IMAGE_AGENT_NODE in executed


def test_research_only_routes_research_agent_to_output_assembler() -> None:
    state = create_initial_state(requested_outputs=["research"])
    assert route_from_query_handler(state) == RESEARCH_AGENT_NODE
    assert route_from_research_agent(state) == OUTPUT_ASSEMBLER_ROUTE_NODE

    compiled = build_langgraph()
    executed = _executed_nodes(compiled, state)
    assert "query_handler_node" in executed
    assert RESEARCH_AGENT_NODE in executed
    assert OUTPUT_ASSEMBLER_NODE in executed


def test_clarification_routes_to_end() -> None:
    assert GRAPH_STRUCTURE[CLARIFICATION_NODE] == [END]
    compiled = build_langgraph()
    executed = _executed_nodes(
        compiled, create_initial_state(clarification_needed=True)
    )
    assert executed == ["query_handler_node", CLARIFICATION_NODE]


def test_retry_router_routes_only_to_writers_or_output_assembler(monkeypatch) -> None:
    allowed = {
        BLOG_WRITER_NODE,
        LINKEDIN_WRITER_NODE,
        IMAGE_AGENT_NODE,
        OUTPUT_ASSEMBLER_NODE,
    }
    assert set(GRAPH_STRUCTURE[RETRY_ROUTER_NODE]).issubset(allowed)

    from contentblitz.agents import quality_validator as quality_validator_module

    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.60}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )

    state = create_initial_state(requested_outputs=["blog"])
    compiled = build_langgraph()
    executed = _executed_nodes(compiled, state)
    assert RETRY_ROUTER_NODE in executed


def test_export_and_error_nodes_route_only_to_end() -> None:
    assert GRAPH_STRUCTURE["export_node"] == [END]
    assert GRAPH_STRUCTURE["error_handler_node"] == [END]


def test_export_path_executes_when_requested() -> None:
    state = create_initial_state(requested_outputs=["image"], export_requested=True)
    compiled = build_langgraph()
    executed = _executed_nodes(compiled, state)
    assert "output_assembler_node" in executed
    assert "export_node" in executed


def test_error_path_executes_when_errors_present() -> None:
    state = create_initial_state(errors=[{"message": "bad"}])
    compiled = build_langgraph()
    executed = _executed_nodes(compiled, state)
    assert executed == ["query_handler_node", "error_handler_node"]


def test_parallel_blog_linkedin_drafts_have_clear_status_and_zero_retries() -> None:
    state = create_initial_state(
        user_query="",
        intent="content_creation",
        requested_outputs=["blog", "linkedin"],
        research_required=True,
        retry_requested=False,
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
        },
    )

    compiled = build_langgraph()
    result = compiled.invoke(state)

    assert result["content_drafts"]["blog"]["version"] >= 1
    assert result["content_drafts"]["linkedin"]["version"] >= 1
    assert result["draft_status"]["blog"] == "complete"
    assert result["draft_status"]["linkedin"] == "complete"
    assert result["cost_controls"]["total_retries_used_this_session"] == 0
    assert result.get("workflow_status") != "blog_draft_complete"
