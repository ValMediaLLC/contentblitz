from copy import deepcopy

from contentblitz.agents import retry_router as retry_router_module
from contentblitz.config import RETRY_POLICY
from contentblitz.state import create_initial_state
from contentblitz.workflow.routing import (
    BLOG_WRITER_NODE,
    LINKEDIN_WRITER_NODE,
    OUTPUT_ASSEMBLER_NODE,
    route_from_retry_router,
)


def _merge_updates(state, updates):
    merged = deepcopy(state)
    merged.update(updates)
    return merged


def _base_state(**overrides):
    state = create_initial_state(
        retry_requested=True,
        quality_scores={},
        retry_feedback={"blog": [], "linkedin": []},
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "max_total_retries_per_session": 3,
            "budget_exceeded": False,
        },
    )
    state.update(overrides)
    return state


def test_blog_retry_increments_blog_writer_before_routing() -> None:
    state = _base_state(
        quality_scores={"blog": {"validation_status": "retry_needed", "composite": 0.62}}
    )
    updates = retry_router_module.retry_router_node(state)
    routed_state = _merge_updates(state, updates)

    assert updates["retry_counts"]["blog_writer"] == 1
    assert updates["cost_controls"]["total_retries_used_this_session"] == 1
    assert route_from_retry_router(routed_state) == BLOG_WRITER_NODE


def test_linkedin_retry_increments_linkedin_writer_before_routing() -> None:
    state = _base_state(
        quality_scores={
            "linkedin": {"validation_status": "retry_needed", "composite": 0.61}
        }
    )
    updates = retry_router_module.retry_router_node(state)
    routed_state = _merge_updates(state, updates)

    assert updates["retry_counts"]["linkedin_writer"] == 1
    assert updates["cost_controls"]["total_retries_used_this_session"] == 1
    assert route_from_retry_router(routed_state) == LINKEDIN_WRITER_NODE


def test_both_retry_fan_out() -> None:
    state = _base_state(
        quality_scores={
            "blog": {"validation_status": "retry_needed", "composite": 0.58},
            "linkedin": {"validation_status": "retry_needed", "composite": 0.59},
        }
    )
    updates = retry_router_module.retry_router_node(state)
    routed_state = _merge_updates(state, updates)
    route = route_from_retry_router(routed_state)

    assert updates["retry_targets"] == ["blog", "linkedin"]
    assert updates["retry_counts"]["blog_writer"] == 1
    assert updates["retry_counts"]["linkedin_writer"] == 1
    assert updates["cost_controls"]["total_retries_used_this_session"] == 2
    assert route == [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE]


def test_max_retry_prevents_dispatch() -> None:
    state = _base_state(
        retry_counts={**create_initial_state()["retry_counts"], "blog_writer": RETRY_POLICY["blog_writer"]},
        quality_scores={"blog": {"validation_status": "retry_needed", "composite": 0.60}},
    )
    updates = retry_router_module.retry_router_node(state)
    routed_state = _merge_updates(state, updates)

    assert updates["retry_requested"] is False
    assert updates["retry_target"] == ""
    assert updates["retry_targets"] == []
    assert updates["_retry_counts_incremented"] is False
    assert route_from_retry_router(routed_state) == OUTPUT_ASSEMBLER_NODE


def test_session_retry_cap_routes_to_output_assembler() -> None:
    state = _base_state(
        quality_scores={"blog": {"validation_status": "retry_needed", "composite": 0.60}},
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 1,
            "max_total_retries_per_session": 1,
            "budget_exceeded": False,
        },
    )
    updates = retry_router_module.retry_router_node(state)
    routed_state = _merge_updates(state, updates)

    assert updates["retry_requested"] is False
    assert route_from_retry_router(routed_state) == OUTPUT_ASSEMBLER_NODE


def test_retry_feedback_populated() -> None:
    state = _base_state(
        quality_scores={"blog": {"validation_status": "retry_needed", "composite": 0.63}}
    )
    updates = retry_router_module.retry_router_node(state)

    feedback_entries = updates["retry_feedback"]["blog"]
    assert len(feedback_entries) == 1
    assert "Retry needed for blog" in feedback_entries[0]
    assert "composite=0.63" in feedback_entries[0]


def test_route_after_retry_router_sees_post_increment_counts() -> None:
    state = _base_state(
        quality_scores={"blog": {"validation_status": "retry_needed", "composite": 0.61}}
    )
    first_updates = retry_router_module.retry_router_node(state)
    first_state = _merge_updates(state, first_updates)
    first_route = route_from_retry_router(first_state)

    second_updates = retry_router_module.retry_router_node(first_state)
    second_state = _merge_updates(first_state, second_updates)
    second_route = route_from_retry_router(second_state)

    assert first_updates["retry_counts"]["blog_writer"] == 1
    assert first_route == BLOG_WRITER_NODE
    assert second_updates["retry_requested"] is False
    assert second_route == OUTPUT_ASSEMBLER_NODE


def test_explicit_retry_target_without_retry_needed_score_does_not_increment() -> None:
    state = _base_state(
        retry_requested=True,
        retry_target="blog",
        quality_scores={},
    )
    updates = retry_router_module.retry_router_node(state)
    assert updates == {}
