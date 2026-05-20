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


def _exports(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("export_outputs") or {}


def _assert_best_or_fallback(result: dict[str, Any], output_type: str) -> None:
    best = (result.get("best_drafts") or {}).get(output_type)
    if best:
        return
    draft = ((result.get("content_drafts") or {}).get(output_type) or {})
    quality = ((result.get("quality_scores") or {}).get(output_type) or {})
    assert bool(draft.get("fallback_generated", False)) is True
    assert bool(draft.get("degraded_generation", False)) is True
    assert str(quality.get("validation_status", "")).strip().lower() == "degraded"


def test_blog_export_is_generated() -> None:
    result = _run_prompt(
        "create a blog article about future AI workflows in marketing agencies"
    )

    exports = _exports(result)

    assert result["workflow_status"] != "failed"
    assert exports.get("blog")

    blog_export = exports["blog"]

    assert blog_export.get("format")
    assert blog_export.get("content")
    assert blog_export.get("filename")

    assert result["assembled_outputs"].get("blog")
    _assert_best_or_fallback(result, "blog")

    assert result["retry_counts"]["blog_writer"] == 0
    assert result["retry_counts"]["linkedin_writer"] == 0

    assert _errors_are_nonfatal(result)


def test_linkedin_export_is_generated() -> None:
    result = _run_prompt("write a linkedin post about AI content marketing trends")

    exports = _exports(result)

    assert exports.get("linkedin")

    linkedin_export = exports["linkedin"]

    assert linkedin_export.get("format")
    assert linkedin_export.get("content")
    assert linkedin_export.get("filename")

    assert result["assembled_outputs"].get("linkedin")
    _assert_best_or_fallback(result, "linkedin")

    assert _errors_are_nonfatal(result)


def test_blog_and_linkedin_exports_are_generated() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content "
        "strategy systems"
    )

    exports = _exports(result)

    assert exports.get("blog")
    assert exports.get("linkedin")

    assert result["assembled_outputs"].get("blog")
    assert result["assembled_outputs"].get("linkedin")

    _assert_best_or_fallback(result, "blog")
    _assert_best_or_fallback(result, "linkedin")

    assert _errors_are_nonfatal(result)


def test_research_export_is_generated() -> None:
    result = _run_prompt("research AI content marketing trends for 2026")

    exports = _exports(result)

    assert exports.get("research")

    research_export = exports["research"]

    assert research_export.get("format")
    assert research_export.get("content")
    assert research_export.get("filename")

    assert result.get("research_data")
    assert result["assembled_outputs"].get("research")

    assert _errors_are_nonfatal(result)


def test_image_export_handles_recoverable_failure_gracefully() -> None:
    result = _run_prompt(
        "create some futuristic images that I can use on clothing designs"
    )

    exports = _exports(result)

    assert exports.get("image")

    image_export = exports["image"]

    assert image_export.get("format")
    assert image_export.get("content")
    assert image_export.get("filename")

    image_outputs = result.get("image_outputs") or []

    if any(output.get("status") == "failed" for output in image_outputs):
        assert (
            "failed" in str(image_export["content"]).lower()
            or "unavailable" in str(image_export["content"]).lower()
        )

    assert _errors_are_nonfatal(result)


def test_blog_and_image_exports_are_generated() -> None:
    result = _run_prompt(
        "create a blog article with futuristic image concepts for AI fashion branding"
    )

    exports = _exports(result)

    assert exports.get("blog")
    assert exports.get("image")

    assert result["assembled_outputs"].get("blog")
    assert result["assembled_outputs"].get("image")

    _assert_best_or_fallback(result, "blog")

    assert _errors_are_nonfatal(result)


def test_export_node_does_not_mutate_best_drafts() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content "
        "strategy systems"
    )

    best_drafts = result.get("best_drafts") or {}
    exports = _exports(result)

    assert exports.get("blog")
    assert exports.get("linkedin")
    _assert_best_or_fallback(result, "blog")
    _assert_best_or_fallback(result, "linkedin")

    if best_drafts.get("blog"):
        assert (
            best_drafts["blog"]["version"]
            == result["content_drafts"]["blog"]["version"]
        )

    if best_drafts.get("linkedin"):
        assert (
            best_drafts["linkedin"]["version"]
            == result["content_drafts"]["linkedin"]["version"]
        )

    assert _errors_are_nonfatal(result)


def test_export_node_does_not_increment_retry_counters() -> None:
    result = _run_prompt(
        "write a blog article and linkedin post about AI-powered content "
        "strategy systems"
    )

    assert result["cost_controls"]["total_retries_used_this_session"] == 0
    assert result["retry_counts"]["blog_writer"] == 0
    assert result["retry_counts"]["linkedin_writer"] == 0

    assert _errors_are_nonfatal(result)


def test_export_node_preserves_assembled_outputs() -> None:
    result = _run_prompt(
        "create a blog article, linkedin post, and image concept about AI "
        "marketing automation"
    )

    assembled = result.get("assembled_outputs") or {}
    exports = _exports(result)

    assert assembled.get("blog")
    assert assembled.get("linkedin")
    assert assembled.get("image")

    assert exports.get("blog")
    assert exports.get("linkedin")
    assert exports.get("image")

    assert _errors_are_nonfatal(result)


def test_export_node_handles_clarification_route_safely() -> None:
    result = _run_prompt("blog")

    assert result["clarification_needed"] is True
    assert result["routing_decision"] == "clarification_node"

    exports = _exports(result)

    assert exports == {} or exports is not None

    assert _errors_are_nonfatal(result)
