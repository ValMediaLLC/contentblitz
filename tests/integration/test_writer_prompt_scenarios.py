from typing import Any

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph


def _run_prompt(prompt: str) -> dict[str, Any]:
    state = create_initial_state(user_query=prompt)
    graph = build_langgraph()
    return graph.invoke(state)


def _draft(result: dict[str, Any], output_type: str) -> dict[str, Any]:
    return (result.get("content_drafts") or {}).get(output_type) or {}


def _draft_body(result: dict[str, Any], output_type: str) -> str:
    return _draft(result, output_type).get("body") or ""


def _errors_are_nonfatal(result: dict[str, Any]) -> bool:
    errors = result.get("errors") or []
    return all(error.get("recoverable") is True for error in errors)


def _assert_blog_generated(result: dict[str, Any]) -> None:
    blog = _draft(result, "blog")

    assert "blog" in result["requested_outputs"]
    assert result["content_brief"]["blog"]
    assert blog.get("version") == 1
    assert blog.get("body")
    assert blog.get("word_count", 0) > 0
    assert blog.get("readability_score") is not None
    assert result["quality_scores"].get("blog")
    assert result["best_drafts"].get("blog")
    assert result["draft_status"].get("blog") == "complete"


def _assert_linkedin_generated(result: dict[str, Any]) -> None:
    linkedin = _draft(result, "linkedin")

    assert "linkedin" in result["requested_outputs"]
    assert result["content_brief"]["linkedin"]
    assert linkedin.get("version") == 1
    assert linkedin.get("body")
    assert linkedin.get("character_count", 0) > 0
    assert linkedin.get("hook")
    assert linkedin.get("cta")
    assert linkedin.get("hashtags")
    assert result["quality_scores"].get("linkedin")
    assert result["best_drafts"].get("linkedin")
    assert result["draft_status"].get("linkedin") == "complete"


def test_blog_only_prompt_generates_only_blog_writer_output() -> None:
    result = _run_prompt(
        "create a blog article about future AI workflows in marketing agencies"
    )

    assert result["requested_outputs"] == ["blog"]
    _assert_blog_generated(result)

    assert _draft_body(result, "linkedin") == ""
    assert not result["quality_scores"].get("linkedin")
    assert result["best_drafts"].get("linkedin") is None
    assert _errors_are_nonfatal(result)


def test_technical_blog_prompt_generates_blog_with_readability_metadata() -> None:
    result = _run_prompt(
        "write a highly technical blog article about multi-agent orchestration using LangGraph and retrieval-augmented generation"
    )

    _assert_blog_generated(result)
    assert _draft(result, "blog").get("readability_score") is not None
    assert _draft(result, "blog").get("word_count", 0) > 0
    assert _errors_are_nonfatal(result)


def test_beginner_friendly_blog_prompt_generates_blog() -> None:
    result = _run_prompt(
        "write a beginner-friendly blog explaining AI workflows for small business owners"
    )

    _assert_blog_generated(result)
    assert _draft_body(result, "blog")
    assert _errors_are_nonfatal(result)


def test_opinionated_blog_prompt_generates_blog() -> None:
    result = _run_prompt(
        "write a controversial blog post arguing that most AI marketing tools are overhyped"
    )

    _assert_blog_generated(result)
    assert _draft_body(result, "blog")
    assert _errors_are_nonfatal(result)


def test_linkedin_prompt_generates_linkedin_writer_output() -> None:
    result = _run_prompt("write a linkedin post about AI content marketing trends")

    _assert_linkedin_generated(result)

    # Known Query Handler TODO:
    # LinkedIn-only prompts may currently also include blog output.
    assert _errors_are_nonfatal(result)


def test_remote_work_linkedin_prompt_generates_linkedin_metadata() -> None:
    result = _run_prompt(
        "create a linkedin post about remote work trends for marketing teams"
    )

    _assert_linkedin_generated(result)

    linkedin = _draft(result, "linkedin")
    assert linkedin.get("character_count", 0) <= 1600
    assert _errors_are_nonfatal(result)


def test_blog_and_linkedin_prompt_generates_both_writer_outputs() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content strategy systems"
    )

    _assert_blog_generated(result)
    _assert_linkedin_generated(result)

    assert result["draft_status"]["blog"] == "complete"
    assert result["draft_status"]["linkedin"] == "complete"
    assert result["quality_scores"].get("blog")
    assert result["quality_scores"].get("linkedin")
    assert _errors_are_nonfatal(result)


def test_weak_blog_prompt_routes_to_clarification_without_drafts() -> None:
    result = _run_prompt("blog")

    assert result["clarification_needed"] is True
    assert result["routing_decision"] == "clarification_node"
    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""
    assert result["quality_scores"] == {}
    assert result["attempt_history"]["blog"] == []
    assert result["attempt_history"]["linkedin"] == []
    assert _errors_are_nonfatal(result)


def test_future_ai_marketing_prompt_does_not_crash() -> None:
    result = _run_prompt("future AI marketing")

    assert _errors_are_nonfatal(result)

    if result.get("clarification_needed"):
        assert _draft_body(result, "blog") == ""
        assert _draft_body(result, "linkedin") == ""
    else:
        assert result["workflow_status"] != "failed"


def test_writer_prompt_does_not_modify_image_outputs() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content strategy systems"
    )

    _assert_blog_generated(result)
    _assert_linkedin_generated(result)

    assert result.get("image_prompts") == []
    assert result.get("image_outputs") == []
    assert result["attempt_history"]["image"] == []
    assert _errors_are_nonfatal(result)
