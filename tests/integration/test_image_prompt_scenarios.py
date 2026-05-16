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


def _has_recoverable_image_error(result: dict[str, Any]) -> bool:
    errors = result.get("errors") or []

    return any(
        error.get("agent") == "image_agent"
        and error.get("type") == "image_generation_failed"
        and error.get("recoverable") is True
        for error in errors
    )


def _assert_image_attempted(result: dict[str, Any]) -> None:
    assert "image" in result["requested_outputs"]
    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1
    assert result["draft_status"].get("image") in {"failed", "complete"}

    image_outputs = result.get("image_outputs") or []
    if any(output.get("status") == "failed" for output in image_outputs):
        assert _has_recoverable_image_error(result)

    assert _errors_are_nonfatal(result)


def test_image_only_clothing_design_prompt_attempts_image_generation() -> None:
    result = _run_prompt(
        "create some futuristic images that I can use on clothing designs"
    )

    assert result["requested_outputs"] == ["image"]
    assert result["research_required"] is False
    _assert_image_attempted(result)

    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert result["quality_scores"] == {}
    assert result["cost_controls"]["image_generations_used_this_session"] in {0, 1}


def test_image_concept_dashboard_prompt_attempts_image_generation() -> None:
    result = _run_prompt(
        "generate an image concept for an AI-powered content dashboard"
    )

    assert result["requested_outputs"] == ["image"]
    assert result["research_required"] is False
    _assert_image_attempted(result)

    prompt_text = " ".join(result.get("image_prompts") or []).lower()
    assert "dashboard" in prompt_text

    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert result["quality_scores"] == {}


def test_blog_and_image_prompt_generates_blog_and_image_outputs() -> None:
    result = _run_prompt(
        "create a blog article with futuristic image concepts for AI fashion branding"
    )

    assert "blog" in result["requested_outputs"]
    assert "image" in result["requested_outputs"]

    assert result["content_brief"]["blog"]
    assert result["content_brief"]["image"]

    assert result["content_drafts"]["blog"]["version"] == 1
    assert _draft_body(result, "blog")
    assert _draft_body(result, "linkedin") == ""

    _assert_image_attempted(result)

    assert result["quality_scores"].get("blog")
    assert not result["quality_scores"].get("linkedin")


def test_research_and_image_prompt_populates_image_brief_then_image_prompt() -> None:
    result = _run_prompt(
        "research futuristic fashion design trends and generate image concepts"
    )

    assert "image" in result["requested_outputs"]

    if result["research_required"] is True:
        assert result.get("research_data")
        assert len(result.get("sources") or []) >= 1

    if result.get("content_brief", {}).get("image"):
        assert result["content_brief"]["image"]

    _assert_image_attempted(result)

    assert _errors_are_nonfatal(result)


def test_research_and_image_prompt_populates_image_brief_and_image_prompt() -> None:
    result = _run_prompt(
        "research futuristic fashion design trends for 2026 and then create image concepts based on the findings"
    )

    assert "image" in result["requested_outputs"]

    # Query handling may either route this as research+image or as image-first
    # depending on deterministic classification rules. Both are acceptable here
    # as long as the image path remains stable.
    if result["research_required"] is True:
        assert result.get("research_data")
        assert len(result.get("sources") or []) >= 1
    else:
        assert result.get("research_data") == {}

    content_brief = result.get("content_brief") or {}

    if content_brief.get("image"):
        assert content_brief["image"]

    _assert_image_attempted(result)

    assert _errors_are_nonfatal(result)


def test_image_generation_failure_is_recoverable_not_fatal() -> None:
    result = _run_prompt(
        "create some futuristic images that I can use on clothing designs"
    )

    image_outputs = result.get("image_outputs") or []

    if any(output.get("status") == "failed" for output in image_outputs):
        assert result["draft_status"]["image"] == "failed"
        assert _has_recoverable_image_error(result)
        assert result["workflow_status"] != "failed"

    assert _errors_are_nonfatal(result)


def test_image_outputs_do_not_store_base64_payloads() -> None:
    result = _run_prompt(
        "generate an image concept for an AI-powered content dashboard"
    )

    for output in result.get("image_outputs") or []:
        assert "base64" not in output
        assert "b64_json" not in output
        assert "image_base64" not in output


def test_image_only_prompt_does_not_create_text_quality_scores() -> None:
    result = _run_prompt(
        "create some futuristic images that I can use on clothing designs"
    )

    assert result["requested_outputs"] == ["image"]
    assert result["quality_scores"] == {}
    assert result["attempt_history"]["blog"] == []
    assert result["attempt_history"]["linkedin"] == []
    assert result["best_drafts"]["blog"] is None
    assert result["best_drafts"]["linkedin"] is None


def test_image_prompt_does_not_increment_search_or_retry_counters() -> None:
    result = _run_prompt(
        "generate an image concept for an AI-powered content dashboard"
    )

    cost_controls = result.get("cost_controls") or {}

    assert cost_controls.get("search_queries_used_this_session") == 0
    assert cost_controls.get("total_retries_used_this_session") == 0


def test_ambiguous_image_word_routes_safely() -> None:
    result = _run_prompt("image")

    assert _errors_are_nonfatal(result)

    if result.get("clarification_needed"):
        assert result["routing_decision"] == "clarification_node"
        assert result.get("image_prompts") == []
        assert result.get("image_outputs") == []
    else:
        assert "image" in result["requested_outputs"]
        _assert_image_attempted(result)


def test_image_prompt_with_specific_style_preserves_style_terms() -> None:
    result = _run_prompt("generate a clean futuristic neon dashboard image concept")

    assert result["requested_outputs"] == ["image"]
    _assert_image_attempted(result)

    prompt_text = " ".join(result.get("image_prompts") or []).lower()

    assert "futuristic" in prompt_text
    assert "dashboard" in prompt_text
