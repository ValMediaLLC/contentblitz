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
    return any(
        error.get("agent") == "image_agent"
        and error.get("type") == "image_generation_failed"
        and error.get("recoverable") is True
        for error in result.get("errors", [])
    )


def _assert_best_or_fallback(result: dict[str, Any], output_type: str) -> None:
    best = (result.get("best_drafts") or {}).get(output_type)
    if best:
        return
    draft = ((result.get("content_drafts") or {}).get(output_type) or {})
    quality = ((result.get("quality_scores") or {}).get(output_type) or {})
    assert bool(draft.get("fallback_generated", False)) is True
    assert bool(draft.get("degraded_generation", False)) is True
    assert str(quality.get("validation_status", "")).strip().lower() == "degraded"


def test_blog_linkedin_image_prompt_routes_all_requested_outputs() -> None:
    result = _run_prompt(
        "create a blog article, linkedin post, and image concept about "
        "AI-powered content strategy systems"
    )

    assert "blog" in result["requested_outputs"]
    assert "linkedin" in result["requested_outputs"]
    assert "image" in result["requested_outputs"]

    assert result["content_brief"]["blog"]
    assert result["content_brief"]["linkedin"]
    assert result["content_brief"]["image"]

    assert result["content_drafts"]["blog"]["version"] == 1
    assert result["content_drafts"]["linkedin"]["version"] == 1
    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1

    assert result["draft_status"]["blog"] == "complete"
    assert result["draft_status"]["linkedin"] == "complete"
    assert result["draft_status"]["image"] in {"failed", "complete"}

    assert result["quality_scores"].get("blog")
    assert result["quality_scores"].get("linkedin")

    if result["draft_status"]["image"] == "failed":
        assert _has_recoverable_image_error(result)

    assert _errors_are_nonfatal(result)


def test_blog_and_image_prompt_does_not_create_linkedin_output() -> None:
    result = _run_prompt(
        "create a blog article with futuristic image concepts for AI fashion branding"
    )

    assert "blog" in result["requested_outputs"]
    assert "image" in result["requested_outputs"]

    assert result["content_drafts"]["blog"]["version"] == 1
    assert _draft_body(result, "linkedin") == ""

    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1

    assert result["quality_scores"].get("blog")
    assert not result["quality_scores"].get("linkedin")

    assert _errors_are_nonfatal(result)


def test_blog_and_linkedin_prompt_does_not_create_image_output() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content "
        "strategy systems"
    )

    assert "blog" in result["requested_outputs"]
    assert "linkedin" in result["requested_outputs"]

    assert result["content_drafts"]["blog"]["version"] == 1
    assert result["content_drafts"]["linkedin"]["version"] == 1

    assert result.get("image_prompts") == []
    assert result.get("image_outputs") == []

    assert result["quality_scores"].get("blog")
    assert result["quality_scores"].get("linkedin")

    assert _errors_are_nonfatal(result)


def test_research_plus_blog_prompt_runs_research_and_blog_without_linkedin() -> None:
    result = _run_prompt(
        "research AI adoption trends in ecommerce marketing and create a blog article"
    )

    assert "blog" in result["requested_outputs"]
    assert result["research_required"] is True
    assert result.get("research_data")
    assert len(result.get("sources") or []) >= 1

    assert result["content_drafts"]["blog"]["version"] == 1
    assert _draft_body(result, "linkedin") == ""

    assert result["quality_scores"].get("blog")
    assert not result["quality_scores"].get("linkedin")

    assert _errors_are_nonfatal(result)


def test_research_plus_linkedin_prompt_runs_research_and_linkedin() -> None:
    result = _run_prompt(
        "research AI adoption trends in ecommerce marketing and create a linkedin post"
    )

    assert "linkedin" in result["requested_outputs"]
    assert result["research_required"] is True
    assert result.get("research_data")
    assert len(result.get("sources") or []) >= 1

    assert result["content_drafts"]["linkedin"]["version"] == 1
    assert result["quality_scores"].get("linkedin")

    assert _errors_are_nonfatal(result)


def test_research_plus_image_prompt_runs_image_path_safely() -> None:
    result = _run_prompt(
        "research futuristic ecommerce branding trends and create an image concept"
    )

    assert "image" in result["requested_outputs"]

    if result["research_required"]:
        assert result.get("research_data")
        assert len(result.get("sources") or []) >= 1

    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1

    if result["draft_status"].get("image") == "failed":
        assert _has_recoverable_image_error(result)

    assert _errors_are_nonfatal(result)


def test_all_output_prompt_preserves_independent_state_sections() -> None:
    result = _run_prompt(
        "create a blog article, linkedin post, and image concept about AI "
        "marketing automation"
    )

    assert "blog" in result["requested_outputs"]
    assert "linkedin" in result["requested_outputs"]
    assert "image" in result["requested_outputs"]

    assert result["content_drafts"]["blog"]["version"] == 1
    assert result["content_drafts"]["linkedin"]["version"] == 1

    assert result["attempt_history"]["blog"]
    assert result["attempt_history"]["linkedin"]

    _assert_best_or_fallback(result, "blog")
    _assert_best_or_fallback(result, "linkedin")

    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1

    assert _errors_are_nonfatal(result)


def test_multi_output_prompt_does_not_increment_retry_counter_on_passed_quality() -> (
    None
):
    result = _run_prompt(
        "create a blog article, linkedin post, and image concept about AI "
        "marketing automation"
    )

    assert result["cost_controls"]["total_retries_used_this_session"] == 0
    assert result["retry_counts"]["blog_writer"] == 0
    assert result["retry_counts"]["linkedin_writer"] == 0

    assert _errors_are_nonfatal(result)
