from __future__ import annotations

import copy

from contentblitz.ui.rendering import (
    build_render_payload,
    dedupe_sources_for_display,
    sanitize_image_outputs_for_display,
)
from contentblitz.ui.status import build_initial_node_statuses


def _base_state() -> dict:
    return {
        "workflow_status": "partial_success",
        "final_response": "Final assembled response.",
        "research_data": {"degraded": False, "synthesized_summary": "Research summary"},
        "content_drafts": {
            "blog": {"body": "Blog draft body"},
            "linkedin": {"body": "LinkedIn draft body"},
            "research_report": {"body": "Research report body"},
        },
        "image_prompts": ["Prompt A"],
        "image_outputs": [{"status": "success", "url": "https://img.example/a.png"}],
        "sources": [],
        "errors": [],
        "quality_scores": {},
        "export_requested": False,
        "export_metadata": {
            "formats_requested": [],
            "export_paths": {},
            "error_log": [],
        },
    }


def test_partial_blog_renders_only_after_blog_writer_completion() -> None:
    state = _base_state()
    statuses = build_initial_node_statuses()
    payload_before = build_render_payload(state=state, node_statuses=statuses)
    assert payload_before["partial_outputs"]["blog"] == ""

    statuses["blog_writer_node"] = "completed"
    payload_after = build_render_payload(state=state, node_statuses=statuses)
    assert payload_after["partial_outputs"]["blog"] == "Blog draft body"
    assert payload_after["partial_output_mode"] == "blog_only"


def test_partial_linkedin_renders_only_after_linkedin_writer_completion() -> None:
    state = _base_state()
    statuses = build_initial_node_statuses()
    statuses["linkedin_writer_node"] = "completed"
    payload = build_render_payload(state=state, node_statuses=statuses)
    assert payload["partial_outputs"]["linkedin"] == "LinkedIn draft body"
    assert payload["partial_output_mode"] == "linkedin_only"


def test_research_report_renders_after_research_or_output_assembler_completion() -> (
    None
):
    state = _base_state()
    statuses = build_initial_node_statuses()
    payload_before = build_render_payload(state=state, node_statuses=statuses)
    assert payload_before["partial_outputs"]["research"] == ""

    statuses["research_agent_node"] = "completed"
    payload_after = build_render_payload(state=state, node_statuses=statuses)
    assert payload_after["partial_outputs"]["research"] == "Research report body"
    assert payload_after["partial_output_mode"] == "research_only"


def test_multi_output_renders_multi_output_mode() -> None:
    state = _base_state()
    statuses = build_initial_node_statuses()
    statuses["blog_writer_node"] = "completed"
    statuses["linkedin_writer_node"] = "completed"
    statuses["output_assembler_node"] = "completed"
    payload = build_render_payload(state=state, node_statuses=statuses)
    assert payload["partial_output_mode"] == "multi_output"
    section_labels = [item["label"] for item in payload["partial_output_sections"]]
    assert "Blog Draft" in section_labels
    assert "LinkedIn Draft" in section_labels


def test_image_only_does_not_render_blog_or_linkedin_partials() -> None:
    state = _base_state()
    state["requested_outputs"] = ["image"]
    statuses = build_initial_node_statuses()
    statuses["image_agent_node"] = "completed"
    payload = build_render_payload(state=state, node_statuses=statuses)
    assert payload["partial_outputs"]["blog"] == ""
    assert payload["partial_outputs"]["linkedin"] == ""
    assert payload["partial_output_mode"] == "none"


def test_final_response_is_included_when_available() -> None:
    state = _base_state()
    payload = build_render_payload(
        state=state, node_statuses=build_initial_node_statuses()
    )
    assert payload["final_response"] == "Final assembled response."


def test_completed_blog_session_derives_display_output_when_partial_mode_is_none() -> (
    None
):
    state = _base_state()
    state["workflow_status"] = "success"
    state["final_response"] = ""
    state["partial_output_mode"] = "none"
    state["partial_outputs"] = {"blog": "", "linkedin": "", "research": ""}
    state["content_drafts"]["blog"]["body"] = "Recovered blog body from content drafts."
    state["ui_node_statuses"] = {"blog_writer_node": "completed"}

    payload = build_render_payload(
        state=state, node_statuses=build_initial_node_statuses()
    )

    assert (
        payload["partial_outputs"]["blog"] == "Recovered blog body from content drafts."
    )
    assert payload["partial_output_mode"] == "blog_only"


def test_image_output_rendering_rejects_base64_content() -> None:
    outputs = [
        {"status": "success", "url": "https://img.example/clean.png"},
        {"status": "success", "url": "data:image/png;base64,ABC"},
        {
            "status": "success",
            "url": "https://img.example/another.png",
            "base64": "ABC",
        },
    ]
    sanitized = sanitize_image_outputs_for_display(outputs)
    assert len(sanitized) == 2
    assert all("base64" not in item for item in sanitized)
    assert all(
        not str(item.get("url", "")).startswith("data:image/") for item in sanitized
    )


def test_image_output_rendering_allows_existing_local_paths(tmp_path) -> None:
    image_file = tmp_path / "local_image.png"
    image_file.write_bytes(b"PNGDATA")

    outputs = [
        {
            "status": "success",
            "provider": "gpt-image-1",
            "local_path": str(image_file),
            "renderable": True,
        }
    ]
    sanitized = sanitize_image_outputs_for_display(outputs)
    assert len(sanitized) == 1
    assert "local_path" in sanitized[0]
    assert sanitized[0]["local_path"].endswith("local_image.png")
    assert sanitized[0]["renderable"] is True


def test_image_output_rendering_sanitizes_text_fields_and_error_payloads() -> None:
    outputs = [
        {
            "status": "success<script>alert(1)</script>",
            "provider": "provider<script>openai</script>",
            "url": "javascript:alert(1)",
            "id": "id<iframe src='https://evil.test'></iframe>",
            "mime_type": "image/png",
            "prompt": "Prompt [x](javascript:alert(1)) OPENAI_API_KEY=sk-secret",
            "revised_prompt": "Revised<object data='x'></object>",
            "width": 1024,
            "height": 1024,
        },
        {
            "status": "failed",
            "provider": "dall-e-3",
            "error": {
                "message": (
                    "{'code': 'configuration_error', 'provider': 'openai', "
                    "'recoverable': False}"
                ),
                "recoverable": True,
            },
        },
    ]

    sanitized = sanitize_image_outputs_for_display(outputs)
    assert len(sanitized) == 2

    success_item = sanitized[0]
    assert "url" not in success_item
    assert success_item["status"] == "success"
    assert success_item["provider"] == "provider"
    assert success_item["id"] == "id"
    assert success_item["mime_type"] == "image/png"
    assert "javascript:" not in success_item["prompt"].lower()
    assert "openai_api_key" not in success_item["prompt"].lower()
    assert "<object" not in success_item["revised_prompt"].lower()

    failed_item = sanitized[1]
    error_message = str(failed_item["error"]["message"]).lower()
    assert "configuration_error" not in error_message
    assert "provider" not in error_message
    assert "openai_api_key" not in error_message


def test_sources_are_deduplicated_for_display() -> None:
    sources = [
        {
            "title": "Source A",
            "url": "https://example.com/a",
            "snippet": "first",
            "citation_available": True,
            "credibility_score": 0.5,
        },
        {
            "title": "Source A Duplicate",
            "url": "https://example.com/a",
            "snippet": "better",
            "citation_available": True,
            "credibility_score": 0.9,
        },
        {
            "title": "Title Duplicate",
            "url": None,
            "snippet": "lower",
            "citation_available": False,
            "credibility_score": 0.2,
        },
        {
            "title": "Title Duplicate",
            "url": None,
            "snippet": "higher",
            "citation_available": False,
            "credibility_score": 0.8,
        },
    ]
    deduped = dedupe_sources_for_display(sources)
    assert len(deduped) == 2
    assert deduped[0]["snippet"] == "better"
    assert deduped[1]["snippet"] == "higher"


def test_render_payload_does_not_mutate_workflow_state() -> None:
    state = _base_state()
    before = copy.deepcopy(state)
    _ = build_render_payload(state=state, node_statuses=build_initial_node_statuses())
    assert state == before


def test_export_warning_without_failed_formats_is_not_non_blocking_failure() -> None:
    state = _base_state()
    state["workflow_status"] = "success"
    state["export_requested"] = True
    state["export_metadata"] = {
        "formats_requested": ["pdf"],
        "export_paths": {"pdf": "exports/content.pdf"},
        "export_status": {"pdf": "completed"},
        "error_log": [{"code": "pdf_validation_warning", "message": "safe warning"}],
        "export_warning_count": 1,
        "export_error_count": 0,
    }
    payload = build_render_payload(
        state=state,
        node_statuses=build_initial_node_statuses(),
    )

    assert payload["export_status"]["export_error_count"] == 0
    assert payload["export_status"]["export_warning_count"] == 1
    assert payload["export_status"]["failed_formats"] == []
    assert payload["export_status"]["non_blocking_failure"] is False


def test_export_off_marks_export_node_skipped_in_payload_statuses() -> None:
    state = _base_state()
    state["export_requested"] = False
    state["export_metadata"] = {
        "formats_requested": [],
        "export_paths": {},
        "error_log": [],
    }
    statuses = build_initial_node_statuses()
    statuses["export_node"] = "completed"
    payload = build_render_payload(state=state, node_statuses=statuses)
    assert payload["node_statuses"]["export_node"] == "skipped"


def test_render_payload_sanitizes_unsafe_final_and_partial_content() -> None:
    state = _base_state()
    state["final_response"] = (
        "Safe intro [bad](javascript:alert(1)) <script>alert(1)</script> "
        "OPENAI_API_KEY=sk-secret"
    )
    state["content_drafts"]["blog"]["body"] = (
        "Blog body with ![img](data:image/png;base64,AAAA) and "
        "<iframe src='https://evil.test'></iframe>"
    )
    statuses = build_initial_node_statuses()
    statuses["blog_writer_node"] = "completed"

    payload = build_render_payload(state=state, node_statuses=statuses)
    lowered_final = payload["final_response"].lower()
    lowered_blog = payload["partial_outputs"]["blog"].lower()

    assert "javascript:" not in lowered_final
    assert "<script" not in lowered_final
    assert "openai_api_key" not in lowered_final
    assert "data:image/" not in lowered_blog
    assert "<iframe" not in lowered_blog
    assert any(
        "unsafe content was removed" in warning.lower()
        for warning in payload["warnings"]
    )


def test_usage_summary_aggregates_safe_counters() -> None:
    state = _base_state()
    state["cost_controls"] = {
        "tokens_used_this_session": 4200,
        "search_queries_used_this_session": 3,
        "image_generations_used_this_session": 1,
        "total_retries_used_this_session": 2,
        "budget_exceeded": False,
    }
    state["retry_counts"] = {"blog_writer": 1, "linkedin_writer": 1}
    state["sources"] = [
        {
            "title": "A",
            "url": "https://a.example",
            "snippet": "x",
            "citation_available": True,
        },
        {
            "title": "B",
            "url": "https://b.example",
            "snippet": "y",
            "citation_available": True,
        },
    ]
    state["image_outputs"] = [
        {"status": "failed", "provider": "dall-e-3"},
        {
            "status": "success",
            "provider": "dall-e-2",
            "url": "https://img.example/a.png",
        },
    ]
    statuses = build_initial_node_statuses()
    statuses["research_agent_node"] = "degraded"

    payload = build_render_payload(state=state, node_statuses=statuses)
    usage = payload["usage_summary"]
    assert usage["estimated_tokens_out"] == 4200
    assert usage["search_queries"] == 3
    assert usage["sources_returned"] == 2
    assert usage["image_generation_requests"] == 2
    assert usage["image_generation_failures"] == 1
    assert usage["retry_attempts"] == 2
    assert usage["degraded_operations"] >= 1
    assert usage["budget_state"] == "degraded"


def test_usage_summary_marks_limited_when_near_caps() -> None:
    state = _base_state()
    state["workflow_status"] = "success"
    state["cost_controls"] = {
        "tokens_used_this_session": 9000,
        "token_budget_per_session": 10000,
        "search_queries_used_this_session": 1,
        "search_query_cap_per_session": 5,
        "image_generations_used_this_session": 0,
        "image_generation_cap_per_session": 3,
        "total_retries_used_this_session": 0,
        "max_total_retries_per_session": 3,
        "budget_exceeded": False,
    }
    state["image_outputs"] = []
    state["warnings"] = []
    statuses = build_initial_node_statuses()
    statuses["research_agent_node"] = "completed"

    payload = build_render_payload(state=state, node_statuses=statuses)
    assert payload["usage_summary"]["budget_state"] == "limited"
    assert any("limited mode" in warning.lower() for warning in payload["warnings"])


def test_usage_summary_marks_budget_exceeded_and_surfaces_warning() -> None:
    state = _base_state()
    state["cost_controls"] = {
        "tokens_used_this_session": 12000,
        "token_budget_per_session": 10000,
        "budget_exceeded": True,
    }
    payload = build_render_payload(
        state=state, node_statuses=build_initial_node_statuses()
    )
    assert payload["usage_summary"]["budget_state"] == "budget_exceeded"
    assert any(
        "usage limits were reached" in warning.lower()
        for warning in payload["warnings"]
    )
