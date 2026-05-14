from __future__ import annotations

from contentblitz.state import create_initial_state
from contentblitz.workflow.routing import (
    BLOG_WRITER_NODE,
    IMAGE_AGENT_NODE,
    LINKEDIN_WRITER_NODE,
    route_from_content_strategist,
)


def test_content_strategist_route_order_is_deterministic_for_full_fanout() -> None:
    state = create_initial_state(requested_outputs=["image", "blog", "linkedin", "blog"])
    route = route_from_content_strategist(state)
    assert route == [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE, IMAGE_AGENT_NODE]


def test_content_strategist_defaults_to_blog_and_linkedin_when_outputs_missing() -> None:
    state = create_initial_state(requested_outputs=[])
    route = route_from_content_strategist(state)
    assert route == [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE]


def test_content_strategist_excludes_unrequested_optional_nodes() -> None:
    state = create_initial_state(requested_outputs=["blog"])
    route = route_from_content_strategist(state)
    assert route == [BLOG_WRITER_NODE]
    assert LINKEDIN_WRITER_NODE not in route
    assert IMAGE_AGENT_NODE not in route


def test_content_strategist_handles_invalid_requested_outputs_shape_deterministically() -> None:
    state = create_initial_state(requested_outputs="not-a-list")  # type: ignore[arg-type]
    route = route_from_content_strategist(state)
    assert route == [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE]
