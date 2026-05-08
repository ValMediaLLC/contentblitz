from typing import Any

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph


def _run_prompt(prompt: str) -> dict[str, Any]:
    state = create_initial_state(user_query=prompt)
    graph = build_langgraph()
    return graph.invoke(state)


def _draft_body(result: dict[str, Any], output_type: str) -> str:
    content_drafts = result.get("content_drafts") or {}
    draft = content_drafts.get(output_type) or {}
    return draft.get("body") or ""


def _errors_are_nonfatal(result: dict[str, Any]) -> bool:
    errors = result.get("errors") or []
    return all(error.get("recoverable") is True for error in errors)


def _assert_clarification_route(result: dict[str, Any]) -> None:
    assert result["clarification_needed"] is True
    assert result["routing_decision"] == "clarification_node"
    assert result["research_required"] is False

    assert result.get("research_data") == {}
    assert result.get("sources") == []

    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""

    assert result.get("image_prompts") == []
    assert result.get("image_outputs") == []

    assert result["quality_scores"] == {}
    assert result["attempt_history"]["blog"] == []
    assert result["attempt_history"]["linkedin"] == []
    assert result["attempt_history"]["image"] == []

    assert _errors_are_nonfatal(result)


def test_short_blog_prompt_routes_to_clarification() -> None:
    result = _run_prompt("blog")
    _assert_clarification_route(result)


def test_short_marketing_prompt_routes_to_clarification() -> None:
    result = _run_prompt("marketing")
    _assert_clarification_route(result)


def test_short_ai_prompt_routes_to_clarification() -> None:
    result = _run_prompt("ai")
    _assert_clarification_route(result)


def test_ai_trends_prompt_routes_to_clarification() -> None:
    result = _run_prompt("AI trends")
    _assert_clarification_route(result)


def test_single_word_linkedin_prompt_routes_to_clarification() -> None:
    result = _run_prompt("linkedin")
    _assert_clarification_route(result)


def test_single_word_image_prompt_routes_safely() -> None:
    result = _run_prompt("image")

    if result.get("clarification_needed"):
        _assert_clarification_route(result)
    else:
        assert "image" in result["requested_outputs"]
        assert len(result.get("image_prompts") or []) >= 1
        assert _errors_are_nonfatal(result)


def test_garbage_prompt_routes_safely() -> None:
    result = _run_prompt("asdfasdfasdf")

    assert _errors_are_nonfatal(result)

    if result.get("clarification_needed"):
        _assert_clarification_route(result)
    else:
        assert result["workflow_status"] != "failed"


def test_empty_prompt_routes_safely() -> None:
    result = _run_prompt("")

    assert _errors_are_nonfatal(result)

    if result.get("clarification_needed"):
        _assert_clarification_route(result)
    else:
        assert result["workflow_status"] != "failed"


def test_whitespace_prompt_routes_safely() -> None:
    result = _run_prompt("   ")

    assert _errors_are_nonfatal(result)

    if result.get("clarification_needed"):
        _assert_clarification_route(result)
    else:
        assert result["workflow_status"] != "failed"


def test_vague_content_prompt_routes_to_clarification_or_safe_fallback() -> None:
    result = _run_prompt("make content")

    assert _errors_are_nonfatal(result)

    if result.get("clarification_needed"):
        _assert_clarification_route(result)
    else:
        assert result["workflow_status"] != "failed"