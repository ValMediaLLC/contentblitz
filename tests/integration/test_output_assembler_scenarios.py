from typing import Any

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph


def _run_prompt(prompt: str) -> dict[str, Any]:
    state = create_initial_state(user_query=prompt)
    graph = build_langgraph()
    return graph.invoke(state)


def _errors_are_nonfatal(result: dict[str, Any]) -> bool:
    errors = result.get("errors") or []
    return all(error.get("recoverable") is True for error in errors)


def _final_response(result: dict[str, Any]) -> str:
    return result.get("final_response") or ""


def _assembled_outputs(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("assembled_outputs") or {}


def test_blog_only_output_is_assembled() -> None:
    result = _run_prompt(
        "create a blog article about future AI workflows in marketing agencies"
    )

    assembled = _assembled_outputs(result)

    assert _final_response(result)
    assert assembled.get("blog")
    assert "linkedin" not in assembled or not assembled.get("linkedin")
    assert result["content_drafts"]["blog"]["version"] == 1
    assert result["best_drafts"].get("blog")
    assert result["cost_controls"]["total_retries_used_this_session"] == 0
    assert _errors_are_nonfatal(result)


def test_blog_and_linkedin_outputs_are_assembled() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content strategy systems"
    )

    assembled = _assembled_outputs(result)

    assert _final_response(result)
    assert assembled.get("blog")
    assert assembled.get("linkedin")
    assert result["content_drafts"]["blog"]["version"] == 1
    assert result["content_drafts"]["linkedin"]["version"] == 1
    assert result["best_drafts"].get("blog")
    assert result["best_drafts"].get("linkedin")
    assert _errors_are_nonfatal(result)


def test_linkedin_included_output_is_assembled() -> None:
    result = _run_prompt("write a linkedin post about AI content marketing trends")

    assembled = _assembled_outputs(result)

    assert _final_response(result)
    assert assembled.get("linkedin")
    assert result["content_drafts"]["linkedin"]["version"] == 1
    assert result["best_drafts"].get("linkedin")
    assert _errors_are_nonfatal(result)


def test_research_only_output_is_assembled_from_research_data() -> None:
    result = _run_prompt("research AI content marketing trends for 2026")

    assembled = _assembled_outputs(result)

    assert _final_response(result)
    assert assembled.get("research")
    assert result.get("research_data")
    assert len(result.get("sources") or []) >= 1
    assert result["content_drafts"]["blog"]["body"] == ""
    assert result["content_drafts"]["linkedin"]["body"] == ""
    assert _errors_are_nonfatal(result)


def test_image_only_recoverable_failure_is_assembled_gracefully() -> None:
    result = _run_prompt("create some futuristic images that I can use on clothing designs")

    assembled = _assembled_outputs(result)

    assert _final_response(result)
    assert assembled.get("image")
    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1

    image_outputs = result.get("image_outputs") or []
    if any(output.get("status") == "failed" for output in image_outputs):
        assert any(output.get("recoverable") is True for output in image_outputs)
        assert "failed" in str(assembled.get("image")).lower() or "unavailable" in str(
            assembled.get("image")
        ).lower()

    assert _errors_are_nonfatal(result)


def test_blog_and_image_outputs_are_assembled() -> None:
    result = _run_prompt(
        "create a blog article with futuristic image concepts for AI fashion branding"
    )

    assembled = _assembled_outputs(result)

    assert _final_response(result)
    assert assembled.get("blog")
    assert assembled.get("image")
    assert result["content_drafts"]["blog"]["version"] == 1
    assert len(result.get("image_prompts") or []) >= 1
    assert len(result.get("image_outputs") or []) >= 1
    assert _errors_are_nonfatal(result)


def test_output_assembler_does_not_mutate_retry_counters_on_normal_path() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content strategy systems"
    )

    assert result["cost_controls"]["total_retries_used_this_session"] == 0
    assert result["retry_counts"]["blog_writer"] == 0
    assert result["retry_counts"]["linkedin_writer"] == 0
    assert _errors_are_nonfatal(result)


def test_output_assembler_preserves_best_drafts() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content strategy systems"
    )

    best_drafts = result.get("best_drafts") or {}

    assert best_drafts.get("blog")
    assert best_drafts.get("linkedin")
    assert best_drafts["blog"]["version"] == result["content_drafts"]["blog"]["version"]
    assert (
        best_drafts["linkedin"]["version"]
        == result["content_drafts"]["linkedin"]["version"]
    )
    assert _errors_are_nonfatal(result)


def test_output_assembler_handles_clarification_path_without_crashing() -> None:
    result = _run_prompt("blog")

    assert result["clarification_needed"] is True
    assert result["routing_decision"] == "clarification_node"
    assert _errors_are_nonfatal(result)