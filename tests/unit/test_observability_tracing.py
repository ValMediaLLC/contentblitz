from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import pytest

from contentblitz.core import observability as observability_module
from contentblitz.workflow.routing import AUTHORITATIVE_NODE_SET


def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


@dataclass
class _SpanFinishRecord:
    metadata: Dict[str, Any]
    outputs: Dict[str, Any]
    error: str


@dataclass
class _RecordingSpan:
    sink: list[dict[str, Any]]
    label: str
    metadata: Dict[str, Any]

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.sink.append(
            {
                "event": f"{self.label}_finish",
                "metadata": dict(metadata or {}),
                "outputs": dict(outputs or {}),
                "error": error.__class__.__name__ if error else "",
            }
        )


@dataclass
class _RecordingTracer:
    events: list[dict[str, Any]] = field(default_factory=list)

    def start_workflow(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> _RecordingSpan:
        self.events.append(
            {"event": "workflow_start", "metadata": dict(metadata), "node_name": ""}
        )
        return _RecordingSpan(
            sink=self.events,
            label="workflow",
            metadata=dict(metadata),
        )

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> _RecordingSpan:
        self.events.append(
            {
                "event": "node_start",
                "metadata": dict(metadata),
                "node_name": str(node_name),
            }
        )
        return _RecordingSpan(
            sink=self.events,
            label="node",
            metadata=dict(metadata),
        )


@dataclass
class _BrokenTracer:
    def start_workflow(self, *, metadata: Mapping[str, Any]) -> Any:
        raise RuntimeError("tracer start_workflow failed")

    def start_node(self, *, node_name: str, metadata: Mapping[str, Any]) -> Any:
        raise RuntimeError("tracer start_node failed")


def test_safe_trace_metadata_contains_only_safe_fields() -> None:
    state = {
        "session_id": "session-123",
        "workflow_status": "partial_success",
        "requested_outputs": ["Blog", "LinkedIn", "image", "blog"],
        "routing_decision": "content_strategist_node",
        "export_requested": True,
        "research_required": True,
        "clarification_needed": False,
        "retry_counts": {"blog_writer": 1, "linkedin_writer": 0, "bad": "x"},
        "cost_controls": {
            "tokens_used_this_session": 9,
            "search_queries_used_this_session": 3,
            "image_generations_used_this_session": 1,
            "total_retries_used_this_session": 1,
            "budget_exceeded": False,
        },
        "export_metadata": {
            "formats_requested": ["PDF", "html"],
            "error_log": [{"format": "pdf", "message": "failed"}],
        },
        "research_data": {"degraded": True},
        "errors": [
            {
                "agent": "image_agent",
                "type": "image_generation_failed",
                "recoverable": True,
            }
        ],
    }

    metadata = observability_module.safe_trace_metadata(
        state,
        node_name="query_handler_node",
        node_status="running",
    )

    assert metadata["session_id"] == "session-123"
    assert metadata["workflow_status"] == "partial_success"
    assert metadata["requested_outputs"] == ["blog", "linkedin", "image"]
    assert metadata["routing_decision"] == "content_strategist_node"
    assert metadata["node_name"] == "query_handler_node"
    assert metadata["node_status"] == "running"
    assert metadata["export_requested"] is True
    assert metadata["research_required"] is True
    assert metadata["clarification_needed"] is False
    assert metadata["degraded_workflow_status"] is True
    assert metadata["recoverable_image_failure_status"] is True
    assert metadata["export_failure_status"] is True
    assert metadata["retry_count_summary"]["blog_writer"] == 1
    assert metadata["cost_counter_summary"]["tokens_used_this_session"] == 9
    assert metadata["export_formats_requested"] == ["pdf", "html"]


def test_safe_trace_metadata_excludes_secrets_and_raw_payloads() -> None:
    state = {
        "session_id": "session-safe",
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "user_query": "full prompt with OPENAI_API_KEY=sk-secret",
        "final_response": "provider response body",
        "tool_outputs": {"raw": {"token": "sk-secret"}},
        "image_outputs": [{"base64": "AAAA", "url": "data:image/png;base64,AAAA"}],
        "errors": [{"message": "traceback with secret sk-secret"}],
        "export_metadata": {"formats_requested": [], "error_log": []},
    }

    metadata = observability_module.safe_trace_metadata(
        state,
        node_name="query_handler_node",
        node_status="completed",
    )
    payload = repr(metadata).lower()

    assert "api_key" not in payload
    assert "sk-secret" not in payload
    assert "user_query" not in payload
    assert "final_response_summary" in metadata
    assert "final_response" not in metadata
    assert "base64" not in payload
    assert "tool_outputs" not in payload


def test_safe_trace_metadata_accepts_only_authoritative_node_names() -> None:
    state = {"requested_outputs": ["blog"]}
    valid = observability_module.safe_trace_metadata(
        state,
        node_name="query_handler_node",
        node_status="running",
    )
    invalid = observability_module.safe_trace_metadata(
        state,
        node_name="unknown_node",
        node_status="running",
    )

    assert valid["node_name"] == "query_handler_node"
    assert "node_name" not in invalid
    assert AUTHORITATIVE_NODE_SET


def test_enabled_tracing_uses_injected_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)
    try:
        tracer = observability_module.get_workflow_tracer()
        workflow_span = tracer.start_workflow(metadata={"requested_outputs": ["blog"]})
        workflow_span.finish(outputs={"workflow_status": "success"})
        node_span = tracer.start_node(
            node_name="query_handler_node",
            metadata={"node_name": "query_handler_node"},
        )
        node_span.finish(outputs={"update_keys": ["routing_decision"]})
    finally:
        observability_module.reset_tracer_factory()

    event_names = [item["event"] for item in recording.events]
    assert "workflow_start" in event_names
    assert "workflow_finish" in event_names
    assert "node_start" in event_names
    assert "node_finish" in event_names


def test_tracer_failures_degrade_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")

    observability_module.set_tracer_factory(lambda _config: _BrokenTracer())
    try:
        tracer = observability_module.get_workflow_tracer()
        workflow_span = tracer.start_workflow(metadata={"requested_outputs": ["blog"]})
        workflow_span.finish(outputs={"workflow_status": "success"})
        node_span = tracer.start_node(
            node_name="query_handler_node",
            metadata={"node_name": "query_handler_node"},
        )
        node_span.finish(outputs={"update_keys": []})
    finally:
        observability_module.reset_tracer_factory()


def test_safe_node_end_metadata_does_not_mutate_input_state() -> None:
    state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
        "export_metadata": {"formats_requested": ["pdf"], "error_log": []},
        "retry_counts": {"blog_writer": 0},
    }
    original_state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
        "export_metadata": {"formats_requested": ["pdf"], "error_log": []},
        "retry_counts": {"blog_writer": 0},
    }
    updates = {
        "workflow_status": "success",
        "retry_counts": {"blog_writer": 1},
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="blog_writer_node",
        node_status="completed",
        updates=updates,
    )

    assert metadata["workflow_status"] == "success"
    assert state == original_state
