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
            {
                "status": "failed",
                "provider": "dall-e-3",
                "prompt": "Create an image concept...",
                "error": {
                    "code": "unknown_error",
                    "message": "{'code': 'configuration_error', 'message': 'OPENAI_API_KEY is not configured.', 'provider': 'openai', 'recoverable': False}",
                    "recoverable": True,
                },
            },
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
                "safe_metadata": {"x": 1, "workflow_status": "success"},
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

    failed_outputs = [
        output for output in serialized["image_outputs"] if output.get("status") == "failed"
    ]
    assert failed_outputs
    first_failed = failed_outputs[0]
    assert first_failed["provider"] == "dall-e-3"
    assert first_failed["prompt"] == "Create an image concept..."
    assert first_failed["error"] == {
        "code": "image_generation_failed",
        "message": "Image generation encountered a recoverable issue.",
        "recoverable": True,
    }
    assert first_failed["error"]["recoverable"] is True
    error_blob = json.dumps(first_failed["error"])
    assert "OPENAI_API_KEY" not in error_blob
    assert "SERP_API_KEY" not in error_blob
    assert "PERPLEXITY_API_KEY" not in error_blob
    assert "configuration_error" not in error_blob
    assert "provider': 'openai'" not in error_blob
    assert "recoverable': False" not in error_blob
    assert all("base64" not in json.dumps(output).lower() for output in serialized["image_outputs"])

    event = serialized["progress_events"][0]
    assert event["safe_metadata"] == {"x": 1}


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


def test_progress_event_metadata_does_not_override_top_level_status() -> None:
    serialized = serialize_workflow_run(
        result_state={
            **_sample_state(),
            "workflow_status": "partial_success",
            "ui_workflow_status": "partial_success",
            "ui_progress_events": [
                {
                    "node_name": "output_assembler_node",
                    "status": "completed",
                    "message": "output_assembler_node completed.",
                    "timestamp": "2026-05-11T04:21:38+00:00",
                    "safe_metadata": {"workflow_status": "success"},
                }
            ],
        },
        run_id="run-progress-1",
        session_id="session-progress-1",
    )

    assert serialized["workflow_status"] == "partial_success"
    assert serialized["ui_workflow_status"] == "partial_success"
    assert serialized["progress_events"][0]["safe_metadata"] == {}

    restored = deserialize_workflow_run(serialized)
    assert restored["workflow_status"] == "partial_success"
    assert restored["ui_workflow_status"] == "partial_success"


def test_export_metadata_status_messages_are_sanitized_on_serialize() -> None:
    state = _sample_state()
    state["export_metadata"] = {
        "formats_requested": ["markdown"],
        "export_paths": {"markdown": "exports/run.md"},
        "export_status": {"markdown": "completed"},
        "status_messages": [
            "OPENAI_API_KEY=sk-super-secret-value",
            "PERPLEXITY_API_KEY=pplx-super-secret-value",
            "Traceback (most recent call last):\nboom",
            "null",
            "",
            "Workflow completed with recoverable warnings.",
        ],
    }

    serialized = serialize_workflow_run(
        result_state=state,
        session_id="session-4",
        run_id="run-4",
    )
    messages = serialized["export_metadata"]["status_messages"]
    blob = json.dumps(messages)

    assert "sk-super-secret-value" not in blob
    assert "pplx-super-secret-value" not in blob
    assert "Traceback (most recent call last)" not in blob
    assert "boom" not in blob
    assert "null" not in [item.lower() for item in messages]
    assert "Workflow completed with recoverable warnings." in messages
    assert any("[REDACTED]" in item for item in messages)
    assert "Internal details were removed." in messages


def test_prompt_injection_metadata_is_serialized_safely() -> None:
    state = _sample_state()
    state["prompt_injection_detected"] = True
    state["prompt_injection_signals"] = [
        "ignore_instructions",
        "REVEAL_SYSTEM_PROMPT",
        "bad-signal!",
        "__proto__",
    ]
    state["sanitized_user_query"] = "Write a blog about AI workflows. OPENAI_API_KEY=sk-danger"

    serialized = serialize_workflow_run(
        result_state=state,
        session_id="session-5",
        run_id="run-5",
    )

    assert serialized["prompt_injection_detected"] is True
    assert serialized["prompt_injection_signals"] == [
        "ignore_instructions",
        "reveal_system_prompt",
        "__proto__",
    ]
    assert "OPENAI_API_KEY" in serialized["sanitized_user_query"]
    assert "[REDACTED]" in serialized["sanitized_user_query"]
    assert "sk-danger" not in serialized["sanitized_user_query"]

    restored = deserialize_workflow_run(serialized)
    assert restored["prompt_injection_detected"] is True
    assert restored["prompt_injection_signals"] == [
        "ignore_instructions",
        "reveal_system_prompt",
        "__proto__",
    ]


def test_prompt_injection_metadata_defaults_when_absent() -> None:
    serialized = serialize_workflow_run(
        result_state=_sample_state(),
        session_id="session-6",
        run_id="run-6",
    )
    restored = deserialize_workflow_run(serialized)

    assert restored["prompt_injection_detected"] is False
    assert restored["prompt_injection_signals"] == []
    assert restored["sanitized_user_query"] == ""


def test_serialization_sanitizes_persisted_drafts_and_final_response_for_unsafe_content() -> None:
    for export_enabled in (True, False):
        state = _sample_state()
        state["final_response"] = (
            "Safe intro <script>alert(1)</script> "
            "[bad](javascript:alert(1)) tail"
        )
        state["content_drafts"]["blog"]["body"] = (
            "Draft body <script>alert(1)</script> "
            "[bad](javascript:alert(1)) end"
        )
        state["export_requested"] = export_enabled
        state["export_metadata"] = {
            "formats_requested": ["markdown"] if export_enabled else [],
            "export_paths": {},
            "export_status": {},
            "status_messages": [],
        }

        serialized = serialize_workflow_run(
            result_state=state,
            session_id=f"session-unsafe-{int(export_enabled)}",
            run_id=f"run-unsafe-{int(export_enabled)}",
        )

        for field_value in (
            serialized["final_response"],
            serialized["content_drafts"]["blog"]["body"],
        ):
            lowered = field_value.lower()
            assert "<script" not in lowered
            assert "javascript:" not in lowered

        restored = deserialize_workflow_run(serialized)
        for field_value in (
            restored["final_response"],
            restored["content_drafts"]["blog"]["body"],
        ):
            lowered = field_value.lower()
            assert "<script" not in lowered
            assert "javascript:" not in lowered


def test_citation_validation_metadata_is_serialized_safely() -> None:
    state = _sample_state()
    state["quality_scores"]["citation_validation"] = {
        "status": "DEGRADED",
        "invalid_count": 2,
        "duplicate_count": 1,
        "unsafe_url_count": 1,
        "missing_count": 1,
        "valid_source_count": 3,
        "raw_payload": {"unsafe": True},
    }

    serialized = serialize_workflow_run(
        result_state=state,
        session_id="session-7",
        run_id="run-7",
    )
    citation = serialized["quality_scores"]["citation_validation"]

    assert citation == {
        "status": "degraded",
        "invalid_count": 2,
        "duplicate_count": 1,
        "unsafe_url_count": 1,
        "missing_count": 1,
        "valid_source_count": 3,
    }

    restored = deserialize_workflow_run(serialized)
    assert restored["quality_scores"]["citation_validation"] == citation


def test_older_record_without_prompt_injection_fields_restores_safely() -> None:
    legacy_record = {
        "run_id": "legacy-1",
        "session_id": "legacy-session",
        "created_at": "2026-05-01T00:00:00+00:00",
        "updated_at": "2026-05-01T00:00:00+00:00",
        "user_query": "Write a blog post",
        "requested_outputs": ["blog"],
        "workflow_status": "success",
        "routing_decision": "content_strategist_node",
        "final_response": "Safe output",
        "content_drafts": {"blog": {"body": "Blog body", "version": 1}},
        "sources": [],
        "quality_scores": {},
        "export_metadata": {"formats_requested": [], "export_paths": {}},
        "warnings": [],
        "errors": [],
        "progress_events": [],
        "status_messages": [],
        "ui_selected_options": {"requested_outputs": ["blog"]},
        "ui_node_statuses": {"query_handler_node": "completed"},
        "ui_workflow_status": "success",
    }

    restored = deserialize_workflow_run(legacy_record)
    assert restored["prompt_injection_detected"] is False
    assert restored["prompt_injection_signals"] == []
    assert restored["sanitized_user_query"] == ""
