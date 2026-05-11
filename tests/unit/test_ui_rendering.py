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
        "export_metadata": {"formats_requested": [], "export_paths": {}, "error_log": []},
    }


def test_partial_blog_renders_only_after_blog_writer_completion() -> None:
    state = _base_state()
    statuses = build_initial_node_statuses()
    payload_before = build_render_payload(state=state, node_statuses=statuses)
    assert payload_before["partial_outputs"]["blog"] == ""

    statuses["blog_writer_node"] = "completed"
    payload_after = build_render_payload(state=state, node_statuses=statuses)
    assert payload_after["partial_outputs"]["blog"] == "Blog draft body"


def test_partial_linkedin_renders_only_after_linkedin_writer_completion() -> None:
    state = _base_state()
    statuses = build_initial_node_statuses()
    statuses["linkedin_writer_node"] = "completed"
    payload = build_render_payload(state=state, node_statuses=statuses)
    assert payload["partial_outputs"]["linkedin"] == "LinkedIn draft body"


def test_research_report_renders_after_research_or_output_assembler_completion() -> None:
    state = _base_state()
    statuses = build_initial_node_statuses()
    payload_before = build_render_payload(state=state, node_statuses=statuses)
    assert payload_before["partial_outputs"]["research"] == ""

    statuses["research_agent_node"] = "completed"
    payload_after = build_render_payload(state=state, node_statuses=statuses)
    assert payload_after["partial_outputs"]["research"] == "Research report body"


def test_final_response_is_included_when_available() -> None:
    state = _base_state()
    payload = build_render_payload(state=state, node_statuses=build_initial_node_statuses())
    assert payload["final_response"] == "Final assembled response."


def test_image_output_rendering_rejects_base64_content() -> None:
    outputs = [
        {"status": "success", "url": "https://img.example/clean.png"},
        {"status": "success", "url": "data:image/png;base64,ABC"},
        {"status": "success", "url": "https://img.example/another.png", "base64": "ABC"},
    ]
    sanitized = sanitize_image_outputs_for_display(outputs)
    assert len(sanitized) == 2
    assert all("base64" not in item for item in sanitized)
    assert all(not str(item.get("url", "")).startswith("data:image/") for item in sanitized)


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
    ]
    deduped = dedupe_sources_for_display(sources)
    assert len(deduped) == 1
    assert deduped[0]["snippet"] == "better"


def test_render_payload_does_not_mutate_workflow_state() -> None:
    state = _base_state()
    before = copy.deepcopy(state)
    _ = build_render_payload(state=state, node_statuses=build_initial_node_statuses())
    assert state == before
