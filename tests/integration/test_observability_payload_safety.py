from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from contentblitz.core import observability as observability_module
from contentblitz.core.redaction import REDACTED_RAW_PAYLOAD


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


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "CONTENTBLITZ_TRACE_SAMPLE_RATE",
        "CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE",
    ):
        monkeypatch.delenv(key, raising=False)


@dataclass
class _RecordingSpan:
    sink: list[dict[str, Any]]
    label: str

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

    def start_workflow(self, *, metadata: Mapping[str, Any]) -> _RecordingSpan:
        self.events.append({"event": "workflow_start", "metadata": dict(metadata)})
        return _RecordingSpan(self.events, "workflow")

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> _RecordingSpan:
        self.events.append(
            {
                "event": "node_start",
                "node_name": str(node_name),
                "metadata": dict(metadata),
            }
        )
        return _RecordingSpan(self.events, str(node_name))

    def start_tool(
        self,
        *,
        tool_name: str,
        metadata: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
    ) -> _RecordingSpan:
        self.events.append(
            {
                "event": "tool_start",
                "tool_name": str(tool_name),
                "metadata": dict(metadata),
                "inputs": dict(inputs or {}),
            }
        )
        return _RecordingSpan(self.events, str(tool_name))


@pytest.fixture(autouse=True)
def _reset_factory() -> None:
    observability_module.reset_tracer_factory()
    yield
    observability_module.reset_tracer_factory()


def test_safe_trace_metadata_summarizes_large_payloads_without_leaking_content(
) -> None:
    very_large_text = "Sensitive generated draft segment. " * 300
    state = {
        "session_id": "session-payload-safety",
        "workflow_status": "partial_success",
        "requested_outputs": ["blog", "linkedin", "image"],
        "routing_decision": "content_strategist_node",
        "user_query": "full raw prompt with OPENAI_API_KEY=sk-super-secret",
        "content_drafts": {
            "blog": {"body": very_large_text, "version": 3},
            "linkedin": {"body": very_large_text, "version": 2},
        },
        "final_response": very_large_text,
        "sources": [
            {
                "title": f"Source {index}",
                "url": f"https://example{index % 5}.com/article/{index}",
                "snippet": "Snippet text",
                "citation_available": index % 2 == 0,
            }
            for index in range(40)
        ],
        "image_outputs": [
            {
                "status": "success",
                "url": "https://img.example/1.png",
                "provider": "openai",
            },
            {"status": "failed", "error": {"recoverable": True}, "provider": "openai"},
        ],
        "cost_controls": {
            "tokens_used_this_session": 200,
            "token_budget_per_session": 1000,
            "search_queries_used_this_session": 4,
            "search_query_cap_per_session": 12,
            "image_generations_used_this_session": 2,
            "image_generation_cap_per_session": 5,
            "total_retries_used_this_session": 1,
            "max_total_retries_per_session": 4,
            "budget_exceeded": False,
        },
    }

    metadata = observability_module.safe_trace_metadata(state)
    metadata_json = json.dumps(metadata, sort_keys=True)
    lowered_payload = metadata_json.lower()

    assert "user_query" not in metadata
    assert "final_response" not in metadata
    assert "content_drafts" not in metadata
    assert "final_response_summary" in metadata
    assert "content_drafts_summary" in metadata
    assert metadata["final_response_summary"]["length"] == len(very_large_text.strip())
    assert metadata["content_drafts_summary"]["blog"]["length"] == len(
        very_large_text.strip()
    )
    assert len(metadata["content_drafts_summary"]["blog"]["preview"]) <= 120
    assert metadata["sources_summary"]["source_count"] == 40
    assert "domains" in metadata["sources_summary"]
    assert metadata["cost_counter_summary"]["token_budget_per_session"] == 1000
    assert "sk-super-secret" not in lowered_payload
    assert very_large_text[:400].lower() not in lowered_payload


def test_workflow_and_node_metadata_use_safe_observability_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "ContentBlitz")
    monkeypatch.setenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com/runs?debug=true",
    )

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)

    tracer = observability_module.get_workflow_tracer()
    workflow_state = {
        "session_id": "session-workflow",
        "workflow_status": "running",
        "requested_outputs": ["blog"],
    }
    workflow_span = tracer.start_workflow(
        metadata=observability_module.safe_workflow_start_metadata(workflow_state)
    )
    node_span = tracer.start_node(
        node_name="query_handler_node",
        metadata=observability_module.safe_node_start_metadata(
            state=workflow_state,
            node_name="query_handler_node",
        ),
    )
    node_span.finish(
        metadata=observability_module.safe_node_end_metadata(
            state=workflow_state,
            node_name="query_handler_node",
            node_status="completed",
        ),
        outputs={"update_keys": ["routing_decision"]},
    )
    workflow_span.finish(
        metadata=observability_module.safe_workflow_end_metadata(
            initial_state=workflow_state,
            final_state={"workflow_status": "success"},
        ),
        outputs={"workflow_status": "success"},
    )

    workflow_start = next(
        event for event in recording.events if event.get("event") == "workflow_start"
    )
    node_start = next(
        event for event in recording.events if event.get("event") == "node_start"
    )
    serialized = json.dumps(recording.events, sort_keys=True)
    all_keys = _collect_metadata_keys(recording.events)

    assert (
        workflow_start["metadata"]["observability_summary"]["tracing_enabled"] is True
    )
    assert (
        workflow_start["metadata"]["observability_summary"]["endpoint_host"]
        == "api.smith.langchain.com"
    )
    assert node_start["metadata"]["observability_summary"]["provider"] == "langsmith"
    assert "LANGSMITH_TRACING" not in all_keys
    assert "LANGSMITH_ENDPOINT" not in all_keys
    assert "LANGSMITH_PROJECT" not in all_keys
    assert "LANGSMITH_API_KEY" not in all_keys
    assert not any(key.upper().endswith("_API_KEY") for key in all_keys)
    assert "https://api.smith.langchain.com/runs?debug=true" not in serialized


def test_tool_span_inputs_redact_raw_payload_keys_and_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE", "1.0")

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)

    span = observability_module.start_tool_span(
        "generate_text",
        metadata={
            "provider": "openai",
            "model": "gpt-4o",
            "agent_key": "blog_writer",
            "raw_prompt": "this should never be included",
        },
        inputs={
            "tool_name": "generate_text",
            "raw_request_payload": '{"prompt":"OPENAI_API_KEY=sk-secret"}',
            "headers": "Authorization: Bearer abc.def.ghi",
        },
    )
    span.finish(metadata={"tool_name": "generate_text"}, outputs={"result_count": 1})

    start_event = next(
        event for event in recording.events if event.get("event") == "tool_start"
    )
    inputs = start_event["inputs"]
    start_metadata = start_event["metadata"]
    flattened = json.dumps(start_event, sort_keys=True).lower()
    all_keys = _collect_metadata_keys(start_event)

    assert inputs["raw_request_payload"] == REDACTED_RAW_PAYLOAD
    assert "sk-secret" not in flattened
    assert "raw_prompt" not in start_metadata
    assert start_metadata["observability_summary"]["provider"] == "langsmith"
    assert "LANGSMITH_TRACING" not in all_keys
    assert "LANGSMITH_ENDPOINT" not in all_keys
    assert "LANGSMITH_PROJECT" not in all_keys
    assert "LANGSMITH_API_KEY" not in all_keys
    assert not any(key.upper().endswith("_API_KEY") for key in all_keys)
