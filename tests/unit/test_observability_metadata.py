from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from contentblitz.agents import query_handler as query_handler_module
from contentblitz.core import observability as observability_module
from contentblitz.core.redaction import REDACTED_STACK_TRACE
from contentblitz.state import create_initial_state


def _collect_metadata_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_collect_metadata_keys(nested))
        return keys
    if isinstance(value, list):
        for nested in value:
            keys.update(_collect_metadata_keys(nested))
    return keys


def test_trace_metadata_excludes_raw_user_query_and_final_response() -> None:
    state = {
        "session_id": "session-xyz",
        "user_query": "full raw user query that should never appear",
        "sanitized_user_query": "sanitized prompt preview should also not appear",
        "final_response": "full generated content that should not be traced",
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "routing_decision": "content_strategist_node",
    }

    metadata = observability_module.safe_trace_metadata(state)
    payload = repr(metadata).lower()

    assert "user_query" not in metadata
    assert "sanitized_user_query" not in metadata
    assert "query_preview" not in metadata
    assert "final_response" not in metadata
    assert "full raw user query" not in payload
    assert "sanitized prompt preview" not in payload


def test_trace_metadata_never_uses_query_preview_keys() -> None:
    state = {
        "sanitized_user_query": "Create a blog and LinkedIn post about AI workflows.",
        "workflow_status": "running",
    }

    metadata = observability_module.safe_trace_metadata(state)

    assert "query_preview" not in metadata
    assert "request_preview" not in metadata


def test_trace_metadata_preserves_safe_schema_fields() -> None:
    state = {
        "session_id": "session-123",
        "workflow_status": "partial_success",
        "requested_outputs": ["blog", "image", "blog"],
        "routing_decision": "content_strategist_node",
        "retry_counts": {"blog_writer": 1, "linkedin_writer": 0},
        "cost_controls": {
            "tokens_used_this_session": 12,
            "search_queries_used_this_session": 3,
            "image_generations_used_this_session": 1,
            "total_retries_used_this_session": 1,
            "budget_exceeded": False,
        },
        "export_requested": True,
        "research_required": True,
        "clarification_needed": False,
        "export_metadata": {
            "formats_requested": ["pdf", "html"],
            "error_log": [{"format": "pdf", "message": "failed"}],
        },
        "research_data": {"degraded": True},
        "sources": [{"title": "a"}, {"title": "b"}, {"title": "c"}],
        "image_outputs": [
            {"status": "success", "url": "https://img.example/1.png"},
            {"status": "failed", "error": {"recoverable": True}},
        ],
        "errors": [{"type": "image_generation_failed", "recoverable": True}],
    }

    metadata = observability_module.safe_trace_metadata(
        state,
        node_name="query_handler_node",
        node_status="running",
    )

    assert metadata["session_id"] == "session-123"
    assert metadata["workflow_status"] == "partial_success"
    assert metadata["requested_outputs"] == ["blog", "image"]
    assert metadata["routing_decision"] == "content_strategist_node"
    assert metadata["node_name"] == "query_handler_node"
    assert metadata["node_status"] == "running"
    assert metadata["retry_count_summary"]["blog_writer"] == 1
    assert metadata["cost_counter_summary"]["tokens_used_this_session"] == 12
    assert metadata["export_formats_requested"] == ["pdf", "html"]
    assert metadata["provider_degraded"] is True
    assert metadata["source_count"] == 3
    assert metadata["image_output_count"] == 2


def test_trace_metadata_normalizes_error_summary_and_redacts_secrets() -> None:
    state = {
        "workflow_status": "failed",
        "requested_outputs": ["blog"],
        "errors": [
            {
                "type": "provider_error",
                "message": "OPENAI_API_KEY=sk-secret",
                "recoverable": True,
            },
            {
                "type": "fatal",
                "message": (
                    "Traceback (most recent call last):\n"
                    '  File "x.py", line 1\n'
                    "ValueError: bad"
                ),
                "recoverable": False,
            },
        ],
    }

    metadata = observability_module.safe_trace_metadata(state)
    payload = repr(metadata).lower()

    assert "error_summary" in metadata
    assert "sk-secret" not in payload
    assert "traceback (most recent call last)" not in payload
    assert "openai_api_key" not in payload


def test_trace_metadata_is_json_serializable() -> None:
    state = {
        "session_id": "session-json",
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "routing_decision": "output_assembler_node",
        "retry_counts": {"blog_writer": 0},
        "cost_controls": {"tokens_used_this_session": 1, "budget_exceeded": False},
        "export_metadata": {"formats_requested": ["markdown"], "error_log": []},
    }

    metadata = observability_module.safe_trace_metadata(state)
    encoded = json.dumps(metadata, sort_keys=True)

    assert isinstance(encoded, str)
    assert '"workflow_status": "success"' in encoded


def test_trace_metadata_degraded_success_normalizes_to_partial_success() -> None:
    state = {
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "export_metadata": {
            "formats_requested": ["pdf"],
            "error_log": [{"format": "pdf", "message": "safe export warning"}],
            "export_status": {"pdf": "failed"},
        },
    }

    metadata = observability_module.safe_trace_metadata(state)

    assert metadata["workflow_status"] == "partial_success"
    assert metadata["degraded_workflow_status"] is True
    assert metadata["export_failure_status"] is True


def test_trace_metadata_success_without_degradation_stays_success() -> None:
    state = {
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "export_metadata": {
            "formats_requested": ["pdf"],
            "error_log": [],
            "export_status": {"pdf": "completed"},
        },
    }

    metadata = observability_module.safe_trace_metadata(state)

    assert metadata["workflow_status"] == "success"
    assert metadata["degraded_workflow_status"] is False
    assert metadata["export_failure_status"] is False


def test_trace_metadata_export_warning_without_failed_formats_is_not_failure() -> None:
    state = {
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "export_metadata": {
            "formats_requested": ["pdf"],
            "export_status": {"pdf": "completed"},
            "error_log": [
                {"code": "pdf_validation_warning", "message": "safe warning"}
            ],
            "export_warning_count": 1,
            "export_error_count": 0,
        },
    }

    metadata = observability_module.safe_trace_metadata(state)

    assert metadata["workflow_status"] == "success"
    assert metadata["export_failure_status"] is False
    assert metadata["degraded_workflow_status"] is False
    assert metadata["requested_export_formats"] == ["pdf"]
    assert metadata["completed_export_formats"] == ["pdf"]
    assert metadata["failed_export_formats"] == []
    assert metadata["export_warning_count"] == 1
    assert metadata["export_error_count"] == 0


def test_safe_node_end_metadata_does_not_mutate_state() -> None:
    state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
        "retry_counts": {"blog_writer": 0},
        "cost_controls": {"tokens_used_this_session": 2, "budget_exceeded": False},
    }
    original = deepcopy(state)
    updates = {"workflow_status": "success", "retry_counts": {"blog_writer": 1}}

    _ = observability_module.safe_node_end_metadata(
        state=state,
        node_name="blog_writer_node",
        node_status="completed",
        updates=updates,
    )

    assert state == original


def test_build_node_timing_metadata_emits_safe_duration_and_timestamps() -> None:
    started_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=375)

    metadata = observability_module.build_node_timing_metadata(
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=None,
    )

    assert metadata["node_started_at"].startswith("2026-05-20T12:00:00")
    assert metadata["node_ended_at"].startswith("2026-05-20T12:00:00.375")
    assert metadata["duration_ms"] == 375


def test_safe_node_end_metadata_includes_timing_and_redacts_error_content() -> None:
    started_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=90)
    state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
    }
    updates = {
        "workflow_status": "partial_success",
        "content_drafts": {
            "blog": {
                "body": "## Fallback Blog Outline\nLimited draft body.",
                "model_used": "gpt-4o",
                "fallback_generated": True,
            }
        },
        "errors": [
            {
                "type": "provider_error",
                "message": (
                    "Traceback (most recent call last): OPENAI_API_KEY=sk-secret"
                ),
                "recoverable": True,
            }
        ],
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="blog_writer_node",
        node_status="degraded",
        updates=updates,
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=90,
    )
    serialized = repr(metadata).lower()

    assert metadata["node_name"] == "blog_writer_node"
    assert metadata["node_status"] == "degraded"
    assert metadata["duration_ms"] == 90
    assert metadata["provider"] == "openai"
    assert metadata["model"] == "gpt-4o"
    assert "provider_latency_ms" not in metadata
    assert "traceback (most recent call last)" not in serialized
    assert "openai_api_key" not in serialized


def test_safe_node_end_metadata_keeps_explicit_provider_latency_only() -> None:
    started_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=200)
    state = {
        "workflow_status": "running",
        "requested_outputs": ["image"],
        "tool_outputs": {"image_agent": {"provider_latency_ms": 75}},
    }
    updates = {
        "workflow_status": "partial_success",
        "tool_outputs": {"image_agent": {"provider_latency_ms": 75}},
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="image_agent_node",
        node_status="degraded",
        updates=updates,
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=200,
    )

    assert metadata["duration_ms"] == 200
    assert metadata["provider_latency_ms"] == 75
    assert metadata["provider_latency_ms"] <= metadata["duration_ms"]
    assert metadata["provider_latency_ms"] != metadata["duration_ms"]


def test_trace_metadata_summarizes_final_response_and_drafts() -> None:
    blog_body = "Blog draft content line.\n" * 120
    final_response = "# Workflow Output\n\n" + ("Final assembled content. " * 160)
    state = {
        "workflow_status": "success",
        "requested_outputs": ["blog", "linkedin"],
        "content_drafts": {
            "blog": {"body": blog_body, "version": 2},
            "linkedin": {"body": "Short linkedin draft", "version": 1},
        },
        "final_response": final_response,
    }

    metadata = observability_module.safe_trace_metadata(state)
    payload = repr(metadata)

    assert "final_response" not in metadata
    assert "content_drafts" not in metadata
    assert metadata["final_response_summary"]["length"] == len(final_response.strip())
    assert metadata["content_drafts_summary"]["blog"]["length"] == len(
        blog_body.strip()
    )
    assert metadata["content_drafts_summary"]["blog"]["sha256_prefix"]
    assert metadata["content_drafts_summary"]["blog"]["preview"] != REDACTED_STACK_TRACE
    assert metadata["final_response_summary"]["preview"] != REDACTED_STACK_TRACE
    assert blog_body[:40] not in payload


def test_trace_metadata_uses_safe_observability_summary_without_env_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com/v1/traces?debug=true",
    )
    monkeypatch.setenv("LANGSMITH_PROJECT", "ContentBlitz")

    metadata = observability_module.safe_trace_metadata(
        {
            "workflow_status": "running",
            "requested_outputs": ["blog"],
        }
    )
    serialized = json.dumps(metadata, sort_keys=True)
    all_keys = _collect_metadata_keys(metadata)
    observability_summary = metadata["observability_summary"]

    assert observability_summary["tracing_enabled"] is True
    assert observability_summary["provider"] == "langsmith"
    assert observability_summary["project_name"] == "ContentBlitz"
    assert observability_summary["endpoint_host"] == "api.smith.langchain.com"
    assert "https://api.smith.langchain.com/v1/traces?debug=true" not in serialized
    assert "LANGSMITH_TRACING" not in all_keys
    assert "LANGSMITH_ENDPOINT" not in all_keys
    assert "LANGSMITH_PROJECT" not in all_keys
    assert "LANGSMITH_API_KEY" not in all_keys
    assert not any(key.upper().endswith("_API_KEY") for key in all_keys)


def test_research_summary_preview_is_not_misclassified_as_stack_trace() -> None:
    state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
        "research_data": {
            "degraded": False,
            "synthesized_summary": (
                "Research summary line one.\n"
                "Second line captures key findings from sources.\n"
                "Third line provides next steps."
            ),
        },
    }

    metadata = observability_module.safe_trace_metadata(state)
    preview = metadata["research_summary"]["summary_preview"]

    assert preview != REDACTED_STACK_TRACE
    assert "Research summary line one." in preview


def test_deterministic_prompt_resolved_outputs_appear_in_trace_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mock_query_handler_llm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"output": "not-json"}

    monkeypatch.setattr(query_handler_module, "generate_text", _mock_query_handler_llm)

    query = "Create a blog article, LinkedIn post, and image concept about AI."
    initial_state = create_initial_state(user_query=query)
    updates = query_handler_module.query_handler_node(initial_state)
    merged_state = dict(initial_state)
    merged_state.update(updates)

    metadata = observability_module.safe_trace_metadata(merged_state)

    assert metadata["requested_outputs"] == ["blog", "linkedin", "image"]
    assert metadata["export_requested"] is False
    assert metadata["clarification_needed"] is False


def test_workflow_trace_inputs_omit_empty_intent() -> None:
    metadata = {
        "requested_outputs": [],
        "export_formats_requested": [],
    }

    inputs = observability_module.safe_workflow_trace_inputs(metadata)

    assert inputs == {}


def test_workflow_trace_inputs_include_intent_when_present() -> None:
    metadata = {
        "requested_outputs": ["blog", "linkedin"],
        "export_formats_requested": ["pdf", "markdown"],
    }

    inputs = observability_module.safe_workflow_trace_inputs(metadata)

    assert inputs["intent"] == ["blog", "linkedin", "pdf", "md"]


def test_workflow_trace_inputs_support_export_metadata_format_fallback() -> None:
    metadata = {
        "requested_outputs": ["blog", "image"],
        "export_metadata": {"formats_requested": ["word", "markdown", "unknown"]},
    }

    inputs = observability_module.safe_workflow_trace_inputs(metadata)

    assert inputs["intent"] == ["blog", "image", "md", "docx"]


def test_workflow_trace_inputs_keep_supported_intent_subset_only() -> None:
    metadata = {
        "requested_outputs": ["blog", "research", "image", "linkedin"],
        "export_formats_requested": ["docx", "html", "word", "unknown"],
    }

    inputs = observability_module.safe_workflow_trace_inputs(metadata)

    assert inputs["intent"] == ["blog", "linkedin", "image", "html", "docx"]


def test_workflow_trace_inputs_do_not_infer_intent_from_prompt_text() -> None:
    metadata = {
        "user_query": "Create a blog article, LinkedIn post, image, and export as PDF.",
        "sanitized_user_query": (
            "Create a blog article, LinkedIn post, image, and export as PDF."
        ),
        "requested_outputs": [],
        "export_formats_requested": [],
    }

    inputs = observability_module.safe_workflow_trace_inputs(metadata)

    assert inputs == {}


def test_workflow_trace_inputs_ignore_unsupported_values() -> None:
    metadata = {
        "requested_outputs": ["research", "unknown", "", "blog"],
        "export_formats_requested": ["csv", "pptx", "", "pdf"],
    }

    inputs = observability_module.safe_workflow_trace_inputs(metadata)

    assert inputs["intent"] == ["blog", "pdf"]


def test_trace_metadata_includes_safe_fallback_degradation_flags() -> None:
    state = {
        "workflow_status": "partial_success",
        "requested_outputs": ["blog", "linkedin", "image"],
        "content_drafts": {
            "blog": {
                "body": "## Fallback Blog Outline\nLimited body.",
                "fallback_generated": True,
                "degraded_generation": True,
                "provider_failure_reason": "quota_exceeded",
            },
            "linkedin": {
                "body": "Fallback LinkedIn draft.",
                "fallback_generated": True,
                "degraded_generation": True,
                "provider_failure_reason": "quota_exceeded",
            },
        },
        "image_outputs": [
            {"status": "failed", "error": {"message": "safe", "recoverable": True}}
        ],
        "status_messages": [
            (
                "Draft unavailable because text generation is currently limited. "
                "Research sources were collected successfully and can be used to "
                "regenerate this section once the provider is available."
            ),
            (
                "OpenAI provider unavailable or quota-limited. "
                "ContentBlitz generated limited fallback outputs."
            ),
        ],
    }

    metadata = observability_module.safe_trace_metadata(state)
    serialized = repr(metadata).lower()

    assert metadata["text_generation_degraded"] is True
    assert metadata["image_generation_degraded"] is True
    assert metadata["fallback_content_used"] is True
    assert metadata["fallback_blog_used"] is True
    assert metadata["fallback_linkedin_used"] is True
    assert metadata["deterministic_research_fallback_used"] is False
    assert metadata["real_generation_succeeded"] is False
    assert metadata["provider_failure_reason"] == "quota_exceeded"
    assert metadata["user_warning_count"] >= 1
    assert "traceback" not in serialized
    assert "openai_api_key" not in serialized


def test_trace_metadata_warning_count_uses_deduped_user_facing_warnings() -> None:
    state = {
        "workflow_status": "partial_success",
        "requested_outputs": ["blog", "linkedin", "image"],
        "content_drafts": {
            "blog": {
                "body": "## Fallback Blog Outline",
                "fallback_generated": True,
                "degraded_generation": True,
            },
            "linkedin": {
                "body": "## Fallback LinkedIn Outline",
                "fallback_generated": True,
                "degraded_generation": True,
            },
        },
        "image_outputs": [{"status": "failed"}],
        "status_messages": [
            (
                "Draft unavailable because text generation is currently limited. "
                "Research sources were collected successfully and can be used to "
                "regenerate this section once the provider is available."
            ),
            (
                "OpenAI provider unavailable or quota-limited. "
                "ContentBlitz generated limited fallback outputs."
            ),
            (
                "Image generation failed in this run, but text outputs may still be "
                "usable."
            ),
        ],
        "warnings": [
            (
                "OpenAI provider unavailable or quota-limited. "
                "ContentBlitz generated limited fallback outputs."
            )
        ],
    }

    metadata = observability_module.safe_trace_metadata(state)

    assert metadata["user_warning_count"] == 3
