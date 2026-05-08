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


def test_blog_only_prompt_generates_blog_draft_and_quality_score() -> None:
    result = _run_prompt(
        "create a blog article about future AI workflows in marketing agencies"
    )

    assert result["requested_outputs"] == ["blog"]
    assert result["research_required"] is True
    assert result["content_brief"]["blog"]
    assert result["content_drafts"]["blog"]["version"] == 1
    assert result["quality_scores"].get("blog")
    assert result["best_drafts"].get("blog")
    assert _draft_body(result, "linkedin") == ""
    assert _errors_are_nonfatal(result)


def test_linkedin_prompt_generates_linkedin_draft_and_quality_score() -> None:
    result = _run_prompt("write a linkedin post about AI content marketing trends")

    assert "linkedin" in result["requested_outputs"]
    assert result["content_brief"]["linkedin"]
    assert result["content_drafts"]["linkedin"]["version"] == 1
    assert result["quality_scores"].get("linkedin")
    assert result["best_drafts"].get("linkedin")
    assert _errors_are_nonfatal(result)


def test_blog_and_linkedin_prompt_generates_both_drafts() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content strategy systems"
    )

    assert "blog" in result["requested_outputs"]
    assert "linkedin" in result["requested_outputs"]
    assert result["content_drafts"]["blog"]["version"] == 1
    assert result["content_drafts"]["linkedin"]["version"] == 1
    assert result["quality_scores"].get("blog")
    assert result["quality_scores"].get("linkedin")
    assert result["draft_status"]["blog"] == "complete"
    assert result["draft_status"]["linkedin"] == "complete"
    assert _errors_are_nonfatal(result)


def test_image_only_prompt_generates_image_prompt_and_recoverable_output() -> None:
    result = _run_prompt("create some futuristic images that I can use on clothing designs")

    assert result["requested_outputs"] == ["image"]
    assert result["research_required"] is False
    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1
    assert result["draft_status"]["image"] in {"failed", "complete"}
    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert result["quality_scores"] == {}

    image_outputs = result.get("image_outputs") or []
    if any(output.get("status") == "failed" for output in image_outputs):
        assert _has_recoverable_image_error(result)


def test_blog_and_image_prompt_generates_blog_and_image_outputs() -> None:
    result = _run_prompt(
        "create a blog article with futuristic image concepts for AI fashion branding"
    )

    assert "blog" in result["requested_outputs"]
    assert "image" in result["requested_outputs"]
    assert result["content_brief"]["blog"]
    assert result["content_brief"]["image"]
    assert result["content_drafts"]["blog"]["version"] == 1
    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1
    assert _draft_body(result, "linkedin") == ""
    assert _errors_are_nonfatal(result)


def test_research_only_prompt_populates_research_without_drafts() -> None:
    result = _run_prompt("research AI content marketing trends for 2026")

    assert result["requested_outputs"] == ["research"]
    assert result["research_required"] is True
    assert result.get("research_data")
    assert len(result.get("sources") or []) >= 1
    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert result["quality_scores"] == {}
    assert _errors_are_nonfatal(result)


def test_ambiguous_short_blog_prompt_routes_to_clarification() -> None:
    result = _run_prompt("blog")

    assert result["clarification_needed"] is True
    assert result["routing_decision"] == "clarification_node"
    assert result.get("research_data") == {}
    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert result["quality_scores"] == {}
    assert _errors_are_nonfatal(result)


def test_ambiguous_topic_prompt_routes_to_clarification_without_crashing() -> None:
    result = _run_prompt("AI trends")

    assert result["clarification_needed"] is True or result["intent"] == "clarification"
    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert _errors_are_nonfatal(result)


def test_garbage_input_prompt_does_not_crash() -> None:
    result = _run_prompt("asdfasdfasdf")

    assert result["clarification_needed"] is True or result["workflow_status"] != "failed"
    assert _errors_are_nonfatal(result)

    if result["clarification_needed"] is True:
        assert _draft_body(result, "blog") == ""
        assert _draft_body(result, "linkedin") == ""


def test_image_concept_prompt_generates_image_prompt_without_text_drafts() -> None:
    result = _run_prompt("generate an image concept for an AI-powered content dashboard")

    assert result["requested_outputs"] == ["image"]
    assert result["research_required"] is False
    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1
    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert result["quality_scores"] == {}

    image_outputs = result.get("image_outputs") or []
    if any(output.get("status") == "failed" for output in image_outputs):
        assert _has_recoverable_image_error(result)