from contentblitz.config import RETRY_POLICY
from contentblitz.core.router import (
    increment_retry_count,
    retry_remaining,
    retry_snapshot,
    route_with_retry,
)
from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph


def test_route_with_retry_initializes_and_increments_counts() -> None:
    state = {}
    first = route_with_retry(
        state=state,
        agent_key="blog_writer",
        retry_node="blog_writer_node",
        exhausted_node="output_assembler_node",
    )
    second = route_with_retry(
        state=state,
        agent_key="blog_writer",
        retry_node="blog_writer_node",
        exhausted_node="output_assembler_node",
    )

    assert first == "blog_writer_node"
    assert second == "output_assembler_node"
    assert state["retry_counts"]["blog_writer"] == RETRY_POLICY["blog_writer"] + 1


def test_retry_snapshot_is_copy_and_retry_remaining_is_deterministic() -> None:
    state = create_initial_state()
    assert retry_remaining(state, "linkedin_writer") is True

    increment_retry_count(state, "linkedin_writer")
    assert retry_remaining(state, "linkedin_writer") is True

    increment_retry_count(state, "linkedin_writer")
    assert retry_remaining(state, "linkedin_writer") is False

    snap = retry_snapshot(state)
    snap["linkedin_writer"] = 999
    assert state["retry_counts"]["linkedin_writer"] != 999


def test_router_builds_and_invokes_graph_for_basic_blog_prompt() -> None:
    graph = build_langgraph()
    assert hasattr(graph, "invoke")

    state = create_initial_state(user_query="write a blog post about AI workflows")
    result = graph.invoke(state)

    assert isinstance(result, dict)
    assert "requested_outputs" in result
    assert "content_drafts" in result
    assert "cost_controls" in result
    assert "blog" in result["requested_outputs"]


def test_router_result_preserves_expected_state_keys() -> None:
    graph = build_langgraph()
    result = graph.invoke(
        create_initial_state(
            user_query="create a blog article and linkedin post about AI strategy"
        )
    )

    expected_keys = set(create_initial_state().keys())
    assert expected_keys.issubset(set(result.keys()))


def test_router_handles_clarification_prompt_safely() -> None:
    graph = build_langgraph()
    result = graph.invoke(create_initial_state(user_query="AI"))

    assert isinstance(result, dict)
    assert result["clarification_needed"] is True
    assert result["routing_decision"] == "clarification_node"
    assert result["retry_counts"]["blog_writer"] == 0
    assert result["retry_counts"]["linkedin_writer"] == 0


def test_router_first_pass_does_not_mutate_retry_counts_unexpectedly() -> None:
    graph = build_langgraph()
    result = graph.invoke(
        create_initial_state(
            user_query="write a blog article and linkedin post about AI-powered strategy systems"
        )
    )

    assert result["retry_counts"]["blog_writer"] == 0
    assert result["retry_counts"]["linkedin_writer"] == 0
    assert result["cost_controls"]["total_retries_used_this_session"] == 0


def test_router_propagates_nonfatal_recoverable_errors_without_crashing() -> None:
    graph = build_langgraph()
    result = graph.invoke(
        create_initial_state(user_query="create an image of a futuristic dashboard")
    )

    assert isinstance(result, dict)
    errors = result.get("errors", [])
    assert any(
        error.get("agent") == "image_agent"
        and error.get("type") == "image_generation_failed"
        and error.get("recoverable") is True
        for error in errors
    )
    assert result["workflow_status"] != "failed"
