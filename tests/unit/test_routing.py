from contentblitz.state import create_initial_state
from contentblitz.workflow.routing import (
    AUTHORITATIVE_NODE_SET,
    BLOG_WRITER_NODE,
    CLARIFICATION_NODE,
    CONTENT_STRATEGIST_NODE,
    ERROR_HANDLER_NODE,
    EXPORT_NODE,
    IMAGE_AGENT_NODE,
    LINKEDIN_WRITER_NODE,
    OUTPUT_ASSEMBLER_NODE,
    RESEARCH_AGENT_NODE,
    RETRY_ROUTER_NODE,
    route_from_content_strategist,
    route_from_output_assembler,
    route_from_quality_validator,
    route_from_query_handler,
    route_from_research_agent,
    route_from_retry_router,
)


def _assert_route_valid(route: str | list[str]) -> None:
    if isinstance(route, str):
        assert route in AUTHORITATIVE_NODE_SET
        return
    assert isinstance(route, list)
    assert len(route) > 0
    assert all(node in AUTHORITATIVE_NODE_SET for node in route)


def test_query_handler_routes_to_clarification() -> None:
    state = create_initial_state(clarification_needed=True)
    route = route_from_query_handler(state)
    assert route == CLARIFICATION_NODE
    _assert_route_valid(route)


def test_query_handler_routes_image_only_path() -> None:
    state = create_initial_state(requested_outputs=["image"])
    route = route_from_query_handler(state)
    assert route == IMAGE_AGENT_NODE
    _assert_route_valid(route)


def test_query_handler_routes_to_research() -> None:
    state = create_initial_state(requested_outputs=["research"])
    route = route_from_query_handler(state)
    assert route == RESEARCH_AGENT_NODE
    _assert_route_valid(route)


def test_query_handler_routes_to_strategist_default() -> None:
    state = create_initial_state(requested_outputs=["blog"])
    route = route_from_query_handler(state)
    assert route == CONTENT_STRATEGIST_NODE
    _assert_route_valid(route)


def test_query_handler_routes_to_error_on_errors() -> None:
    state = create_initial_state(errors=[{"message": "boom"}])
    route = route_from_query_handler(state)
    assert route == ERROR_HANDLER_NODE
    _assert_route_valid(route)


def test_research_agent_routes_research_only_to_output_assembler() -> None:
    state = create_initial_state(requested_outputs=["research"])
    route = route_from_research_agent(state)
    assert route == OUTPUT_ASSEMBLER_NODE
    _assert_route_valid(route)


def test_research_agent_routes_mixed_to_strategist() -> None:
    state = create_initial_state(requested_outputs=["research", "blog"])
    route = route_from_research_agent(state)
    assert route == CONTENT_STRATEGIST_NODE
    _assert_route_valid(route)


def test_content_strategist_routes_parallel_writers() -> None:
    state = create_initial_state(requested_outputs=["blog", "linkedin", "image"])
    route = route_from_content_strategist(state)
    assert route == [BLOG_WRITER_NODE, LINKEDIN_WRITER_NODE, IMAGE_AGENT_NODE]
    _assert_route_valid(route)


def test_quality_validator_routes_retry_or_output_or_error() -> None:
    retry_state = create_initial_state(retry_requested=True)
    output_state = create_initial_state()
    error_state = create_initial_state(errors=[{"message": "bad"}])

    retry_route = route_from_quality_validator(retry_state)
    output_route = route_from_quality_validator(output_state)
    error_route = route_from_quality_validator(error_state)

    assert retry_route == RETRY_ROUTER_NODE
    assert output_route == OUTPUT_ASSEMBLER_NODE
    assert error_route == ERROR_HANDLER_NODE
    _assert_route_valid(retry_route)
    _assert_route_valid(output_route)
    _assert_route_valid(error_route)


def test_retry_router_respects_retry_ordering_rule() -> None:
    state = create_initial_state(retry_requested=True, retry_target="blog")
    first = route_from_retry_router(state)
    second = route_from_retry_router(state)

    assert first == BLOG_WRITER_NODE
    assert second == BLOG_WRITER_NODE
    assert state["retry_counts"]["blog_writer"] == 0
    _assert_route_valid(first)
    _assert_route_valid(second)


def test_retry_router_route_function_does_not_increment_counters() -> None:
    state = create_initial_state(retry_requested=True, retry_target="linkedin")
    before = dict(state["retry_counts"])
    route = route_from_retry_router(state)
    assert route == LINKEDIN_WRITER_NODE
    assert state["retry_counts"] == before


def test_retry_router_routes_only_writer_nodes_or_output_assembler() -> None:
    state = create_initial_state(retry_target="linkedin")
    route = route_from_retry_router(state)
    assert route in {
        BLOG_WRITER_NODE,
        LINKEDIN_WRITER_NODE,
        IMAGE_AGENT_NODE,
        OUTPUT_ASSEMBLER_NODE,
    }
    _assert_route_valid(route)


def test_output_assembler_routes_export_or_default_or_error() -> None:
    export_state = create_initial_state(export_requested=True)
    default_state = create_initial_state()
    error_state = create_initial_state(errors=[{"message": "bad"}])

    export_route = route_from_output_assembler(export_state)
    default_route = route_from_output_assembler(default_state)
    error_route = route_from_output_assembler(error_state)

    assert export_route == EXPORT_NODE
    assert default_route == OUTPUT_ASSEMBLER_NODE
    assert error_route == ERROR_HANDLER_NODE
    _assert_route_valid(export_route)
    _assert_route_valid(default_route)
    _assert_route_valid(error_route)


def test_all_routing_functions_return_non_none_and_authoritative_nodes() -> None:
    checks = [
        route_from_query_handler(create_initial_state()),
        route_from_research_agent(
            create_initial_state(requested_outputs=["research", "blog"])
        ),
        route_from_content_strategist(
            create_initial_state(requested_outputs=["blog", "linkedin"])
        ),
        route_from_quality_validator(create_initial_state()),
        route_from_retry_router(create_initial_state(retry_target="image")),
        route_from_output_assembler(create_initial_state()),
    ]
    for route in checks:
        assert route is not None
        _assert_route_valid(route)
