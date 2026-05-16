from __future__ import annotations

import pytest

from contentblitz.agents import image_agent as image_agent_module
from contentblitz.agents import query_handler as query_handler_module
from contentblitz.ui.progress import create_progress_event
from contentblitz.ui.rendering import build_render_payload
from contentblitz.ui.status import (
    apply_optional_node_skips,
    build_status_messages,
    derive_node_statuses,
    summarize_workflow_status,
)
from frontend.services.orchestrator_client import stream_workflow_progress


def test_mocked_full_workflow_progress_path_renders_partial_success_safely() -> None:
    events = [
        create_progress_event(
            node_name="query_handler_node",
            status="running",
            timestamp="2026-05-10T12:00:00+00:00",
        ),
        create_progress_event(
            node_name="query_handler_node",
            status="completed",
            timestamp="2026-05-10T12:00:01+00:00",
        ),
        create_progress_event(
            node_name="research_agent_node",
            status="degraded",
            timestamp="2026-05-10T12:00:02+00:00",
        ),
        create_progress_event(
            node_name="content_strategist_node",
            status="completed",
            timestamp="2026-05-10T12:00:03+00:00",
        ),
        create_progress_event(
            node_name="blog_writer_node",
            status="completed",
            timestamp="2026-05-10T12:00:04+00:00",
        ),
        create_progress_event(
            node_name="linkedin_writer_node",
            status="completed",
            timestamp="2026-05-10T12:00:05+00:00",
        ),
        create_progress_event(
            node_name="image_agent_node",
            status="degraded",
            timestamp="2026-05-10T12:00:06+00:00",
        ),
        create_progress_event(
            node_name="quality_validator_node",
            status="completed",
            timestamp="2026-05-10T12:00:07+00:00",
        ),
        create_progress_event(
            node_name="retry_router_node",
            status="skipped",
            timestamp="2026-05-10T12:00:08+00:00",
        ),
        create_progress_event(
            node_name="output_assembler_node",
            status="completed",
            timestamp="2026-05-10T12:00:09+00:00",
        ),
        create_progress_event(
            node_name="export_node",
            status="degraded",
            timestamp="2026-05-10T12:00:10+00:00",
        ),
    ]
    node_statuses = derive_node_statuses(events)

    state = {
        "workflow_status": "partial_success",
        "final_response": "Final response exists despite warnings.",
        "research_data": {
            "degraded": True,
            "synthesized_summary": "Degraded research summary",
        },
        "content_drafts": {
            "blog": {"body": "Partial blog draft"},
            "linkedin": {"body": "Partial linkedin draft"},
            "research_report": {"body": "Research report content"},
        },
        "image_prompts": ["Image concept prompt"],
        "image_outputs": [{"status": "failed", "error": "safe error"}],
        "sources": [
            {
                "title": "Source 1",
                "url": "https://example.com/a",
                "snippet": "snippet a",
                "citation_available": True,
                "credibility_score": 0.7,
            },
            {
                "title": "Source 1 duplicate",
                "url": "https://example.com/a",
                "snippet": "better snippet",
                "citation_available": True,
                "credibility_score": 0.9,
            },
            {
                "title": "Source 2 no url",
                "url": None,
                "snippet": "snippet b",
                "citation_available": False,
                "credibility_score": 0.4,
            },
        ],
        "errors": [
            {
                "agent": "image_agent",
                "message": "recoverable failure",
                "recoverable": True,
            },
            {
                "message": "Traceback (most recent call last):\n File 'x.py' line 1",
                "recoverable": False,
            },
        ],
        "quality_scores": {"blog": {"validation_status": "retry_needed"}},
        "export_requested": True,
        "export_metadata": {
            "formats_requested": ["pdf"],
            "export_paths": {"markdown": "exports/content.md"},
            "error_log": [{"message": "PDF export failed: OPENAI_API_KEY=sk-secret"}],
        },
        "cost_controls": {
            "total_retries_used_this_session": 1,
            "budget_exceeded": False,
        },
    }

    render_payload = build_render_payload(state=state, node_statuses=node_statuses)
    messages = build_status_messages(state=state, node_statuses=node_statuses)
    summary = summarize_workflow_status(
        node_statuses, workflow_status=state["workflow_status"]
    )

    assert summary == "partial_success"
    assert node_statuses["query_handler_node"] == "completed"
    assert node_statuses["research_agent_node"] == "degraded"
    assert node_statuses["retry_router_node"] == "skipped"
    assert node_statuses["output_assembler_node"] == "completed"
    assert node_statuses["export_node"] == "degraded"

    assert render_payload["final_response"]
    assert render_payload["partial_outputs"]["blog"] == "Partial blog draft"
    assert render_payload["partial_outputs"]["linkedin"] == "Partial linkedin draft"
    assert render_payload["partial_outputs"]["research"] == "Research report content"
    assert render_payload["export_status"]["non_blocking_failure"] is True
    assert len(render_payload["sources"]) == 2
    assert all("Traceback" not in item["message"] for item in render_payload["errors"])
    assert all("sk-secret" not in item["message"] for item in render_payload["errors"])
    assert any("degraded" in warning.lower() for warning in render_payload["warnings"])
    assert any(
        "Image generation encountered a recoverable issue" in message
        for message in messages
    )


@pytest.mark.parametrize(
    "query",
    [
        "create futuristic cyberpunk hoodie artwork for streetwear branding",
        "create futuristic shark-themed beachwear image concepts",
    ],
)
def test_image_only_routing_and_rendering_remain_deterministic(
    query: str, monkeypatch
) -> None:
    def fail_query_classification(*args, **kwargs):
        raise AssertionError(
            "query_handler.generate_text should not run for explicit image-only requests."
        )

    def fake_prompt_enhancer(*args, **kwargs):
        return {"output": "Enhanced image prompt", "total_tokens": 0}

    def fake_image_provider(*args, **kwargs):
        return {
            "images": [],
            "provider_primary": "dall-e-3",
            "provider_fallback": "dall-e-2",
            "provider_used": "dall-e-2",
            "degraded": True,
            "error": {"code": "provider_failure"},
        }

    monkeypatch.setattr(
        query_handler_module, "generate_text", fail_query_classification
    )
    monkeypatch.setattr(image_agent_module, "generate_text", fake_prompt_enhancer)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_image_provider)

    events: list[dict] = []
    final_result: dict = {}
    for event in stream_workflow_progress(
        user_query=query,
        requested_outputs=["image"],
        export_requested=False,
        export_formats=[],
    ):
        if event.get("type") == "progress":
            raw = event.get("event")
            if isinstance(raw, dict):
                events.append(raw)
        elif event.get("type") == "final":
            payload = event.get("result")
            if isinstance(payload, dict):
                final_result = payload

    node_statuses = apply_optional_node_skips(
        state=final_result,
        node_statuses=derive_node_statuses(events),
    )
    render_payload = build_render_payload(
        state=final_result, node_statuses=node_statuses
    )
    summary = summarize_workflow_status(
        node_statuses,
        workflow_status=str(final_result.get("workflow_status", "")),
    )
    executed_nodes = {str(event.get("node_name", "")).strip() for event in events}

    assert "image_agent_node" in executed_nodes
    assert "research_agent_node" not in executed_nodes
    assert "blog_writer_node" not in executed_nodes
    assert "linkedin_writer_node" not in executed_nodes

    assert node_statuses["blog_writer_node"] in {"pending", "skipped"}
    assert node_statuses["linkedin_writer_node"] in {"pending", "skipped"}
    assert node_statuses["export_node"] == "skipped"

    assert summary == "partial_success"
    assert render_payload["partial_outputs"]["blog"] == ""
    assert render_payload["partial_outputs"]["linkedin"] == ""
    assert render_payload["image_prompts"]
    assert any(
        "recoverable issue" in msg.lower()
        for msg in build_status_messages(
            state=final_result, node_statuses=node_statuses
        )
    )
