from contentblitz.state import create_initial_state
from contentblitz.agents.retry_router import retry_router_node
from contentblitz.workflow.routing import route_from_retry_router
from contentblitz.config import RETRY_POLICY


def _run_retry_scenario(state: dict) -> tuple[dict, str | list[str]]:
    updates = retry_router_node(state)
    merged = {**state, **updates}
    route = route_from_retry_router(merged)
    return merged, route


def test_blog_retry_needed_routes_to_blog_writer() -> None:
    state = create_initial_state(
        user_query="create a blog article about AI workflow automation",
        requested_outputs=["blog"],
    )

    state["quality_scores"] = {
        "blog": {
            "composite": 0.62,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve clarity and add stronger structure."],
        }
    }

    merged, route = _run_retry_scenario(state)

    assert route == "blog_writer_node"
    assert merged["retry_counts"]["blog_writer"] == 1
    assert merged["retry_counts"]["linkedin_writer"] == 0
    assert merged["cost_controls"]["total_retries_used_this_session"] == 1
    assert merged["retry_feedback"]["blog"]
    assert not merged["retry_feedback"]["linkedin"]


def test_linkedin_retry_needed_routes_to_linkedin_writer() -> None:
    state = create_initial_state(
        user_query="write a linkedin post about AI marketing systems",
        requested_outputs=["linkedin"],
    )

    state["quality_scores"] = {
        "linkedin": {
            "composite": 0.64,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Tighten the hook and improve CTA clarity."],
        }
    }

    merged, route = _run_retry_scenario(state)

    assert route == "linkedin_writer_node"
    assert merged["retry_counts"]["linkedin_writer"] == 1
    assert merged["retry_counts"]["blog_writer"] == 0
    assert merged["cost_controls"]["total_retries_used_this_session"] == 1
    assert merged["retry_feedback"]["linkedin"]
    assert not merged["retry_feedback"]["blog"]


def test_blog_and_linkedin_retry_needed_fans_out_to_both_writers() -> None:
    state = create_initial_state(
        user_query="write a blog article and linkedin post about AI systems",
        requested_outputs=["blog", "linkedin"],
    )

    state["quality_scores"] = {
        "blog": {
            "composite": 0.61,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve structure and add clearer takeaways."],
        },
        "linkedin": {
            "composite": 0.63,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve hook and make CTA more specific."],
        },
    }

    merged, route = _run_retry_scenario(state)

    assert route == ["blog_writer_node", "linkedin_writer_node"]
    assert merged["retry_counts"]["blog_writer"] == 1
    assert merged["retry_counts"]["linkedin_writer"] == 1
    assert merged["cost_controls"]["total_retries_used_this_session"] == 2
    assert merged["retry_feedback"]["blog"]
    assert merged["retry_feedback"]["linkedin"]


def test_blog_max_retry_reached_routes_to_output_assembler() -> None:
    state = create_initial_state(
        user_query="create a blog article about AI workflow automation",
        requested_outputs=["blog"],
    )

    state["retry_counts"]["blog_writer"] = RETRY_POLICY["blog_writer"]
    state["quality_scores"] = {
        "blog": {
            "composite": 0.62,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve clarity and add stronger structure."],
        }
    }

    merged, route = _run_retry_scenario(state)

    assert route == "output_assembler_node"
    assert merged["retry_counts"]["blog_writer"] == RETRY_POLICY["blog_writer"]
    assert merged["cost_controls"]["total_retries_used_this_session"] == 0
    assert not merged["retry_feedback"]["blog"]


def test_session_retry_cap_reached_routes_to_output_assembler() -> None:
    state = create_initial_state(
        user_query="create a blog article about AI workflow automation",
        requested_outputs=["blog"],
    )

    state["cost_controls"]["total_retries_used_this_session"] = state[
        "cost_controls"
    ].get("max_total_retries_per_session", 3)

    state["quality_scores"] = {
        "blog": {
            "composite": 0.62,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve clarity and add stronger structure."],
        }
    }

    merged, route = _run_retry_scenario(state)

    assert route == "output_assembler_node"
    assert merged["retry_counts"]["blog_writer"] == 0
    assert merged["cost_controls"]["total_retries_used_this_session"] == 3
    assert not merged["retry_feedback"]["blog"]


def test_passed_quality_scores_do_not_retry() -> None:
    state = create_initial_state(
        user_query="create a blog article about AI workflow automation",
        requested_outputs=["blog"],
    )

    state["quality_scores"] = {
        "blog": {
            "composite": 0.8,
            "passed": True,
            "validation_status": "passed",
        }
    }

    merged, route = _run_retry_scenario(state)

    assert route == "output_assembler_node"
    assert merged["retry_counts"]["blog_writer"] == 0
    assert merged["cost_controls"]["total_retries_used_this_session"] == 0
    assert not merged["retry_feedback"]["blog"]
