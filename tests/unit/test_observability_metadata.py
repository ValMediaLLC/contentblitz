from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from contentblitz.core import observability as observability_module
from contentblitz.core.redaction import REDACTED_STACK_TRACE


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
        "final_response": "full generated content that should not be traced",
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "routing_decision": "content_strategist_node",
    }

    metadata = observability_module.safe_trace_metadata(state)

    assert "user_query" not in metadata
    assert "final_response" not in metadata


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
