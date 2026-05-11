from __future__ import annotations

import json
from pathlib import Path

from contentblitz.persistence.serialization import (
    deserialize_workflow_run,
    serialize_workflow_run,
    to_run_summary,
)


def _sample_state() -> dict:
    return {
        "user_query": "Write a blog on AI workflows.",
        "requested_outputs": ["blog", "research"],
        "workflow_status": "partial_success",
        "ui_workflow_status": "partial_success",
        "routing_decision": "research_agent_node",
        "final_response": "Result body with OPENAI_API_KEY=sk-live-secret-value",
        "content_drafts": {
            "blog": {"body": "Blog body text", "version": 2},
            "linkedin": {"body": "LinkedIn body", "version": 1},
            "research_report": {"body": "Research report body"},
        },
        "partial_outputs": {
            "blog": "Blog partial",
            "linkedin": "",
            "research": "Research partial",
        },
        "partial_output_mode": "multi_output",
        "image_prompts": ["Create a concept art prompt"],
        "image_outputs": [
            {"status": "success", "url": "https://img.example/safe.png"},
            {"status": "failed", "url": "data:image/png;base64,AAAA", "b64_json": "AAAA"},
        ],
        "sources": [
            {
                "title": "Source A",
                "url": "https://example.com/a",
                "snippet": "Details",
                "source": "serp",
                "published_at": "2026-05-01",
                "citation_available": True,
                "credibility_score": 0.9,
                "raw_payload": {"secret": "x"},
            }
        ],
        "quality_scores": {
            "blog": {"validation_status": "retry_needed", "composite": 0.71, "internal": "x"}
        },
        "export_metadata": {
            "formats_requested": ["markdown", "pdf"],
            "export_paths": {"markdown": "exports/run.md", "pdf": "exports/run.pdf"},
            "error_log": [{"message": "Traceback (most recent call last):\nsecret data"}],
        },
        "errors": [
            {
                "message": "Traceback (most recent call last):\nOPENAI_API_KEY=sk-secret",
                "recoverable": False,
            }
        ],
        "warnings": ["PERPLEXITY_API_KEY=pplx-secret"],
        "ui_node_statuses": {"query_handler_node": "completed"},
        "ui_progress_events": [
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "message": "done",
                "timestamp": "2026-05-10T10:00:00+00:00",
                "safe_metadata": {"x": 1},
            }
        ],
        "status_messages": ["SERP_API_KEY=serp-secret", "Workflow completed."],
        "tool_outputs": {"unsafe": "should-not-persist"},
        "cache_metadata": {"unsafe": "should-not-persist"},
    }


def test_serialization_persists_safe_fields_only() -> None:
    serialized = serialize_workflow_run(
        result_state=_sample_state(),
        ui_selected_options={
            "requested_outputs": ["blog", "research"],
            "export_requested": True,
            "export_formats": ["markdown"],
        },
        progress_events=[],
        status_messages=["Workflow completed."],
        session_id="session-1",
        run_id="run-1",
        created_at="2026-05-10T10:00:00+00:00",
    )

    assert serialized["run_id"] == "run-1"
    assert serialized["session_id"] == "session-1"
    assert serialized["requested_outputs"] == ["blog", "research"]
    assert "tool_outputs" not in serialized
    assert "cache_metadata" not in serialized

    blob = json.dumps(serialized)
    assert "sk-live-secret-value" not in blob
    assert "serp-secret" not in blob
    assert "pplx-secret" not in blob
    assert "Traceback (most recent call last)" not in blob
    assert "data:image/png;base64" not in blob
    assert "b64_json" not in blob


def test_deserialization_handles_missing_export_files_safely(tmp_path: Path) -> None:
    existing_export = tmp_path / "existing.md"
    existing_export.write_text("ok", encoding="utf-8")
    serialized = serialize_workflow_run(
        result_state={
            **_sample_state(),
            "export_metadata": {
                "formats_requested": ["markdown", "pdf"],
                "export_paths": {
                    "markdown": str(existing_export),
                    "pdf": str(tmp_path / "missing.pdf"),
                },
            },
        },
        session_id="session-2",
        run_id="run-2",
    )

    restored = deserialize_workflow_run(serialized)
    assert restored["export_metadata"]["export_paths"] == {"markdown": str(existing_export)}
    assert any("missing locally" in warning.lower() for warning in restored["warnings"])


def test_to_run_summary_is_stable() -> None:
    serialized = serialize_workflow_run(
        result_state=_sample_state(),
        session_id="session-3",
        run_id="run-3",
    )
    summary = to_run_summary(serialized)
    assert summary["run_id"] == "run-3"
    assert summary["session_id"] == "session-3"
    assert summary["workflow_status"] == "partial_success"
    assert isinstance(summary["requested_outputs"], list)
