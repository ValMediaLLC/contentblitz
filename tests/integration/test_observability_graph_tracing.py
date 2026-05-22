from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from contentblitz.core import observability as observability_module
from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph
from contentblitz.workflow.routing import AUTHORITATIVE_NODE_SET

pytestmark = pytest.mark.filterwarnings(
    "ignore:The default value of `allowed_objects` will change"
)


def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


@dataclass
class _RecordingSpan:
    sink: list[dict[str, Any]]
    event_name: str

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.sink.append(
            {
                "event": self.event_name,
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
        return _RecordingSpan(self.events, "workflow_finish")

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
        return _RecordingSpan(self.events, "node_finish")


@dataclass
class _FailingTracer:
    def start_workflow(self, *, metadata: Mapping[str, Any]) -> Any:
        raise RuntimeError("workflow tracer failed")

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> Any:
        raise RuntimeError("node tracer failed")


def _invoke_clarification_graph() -> dict[str, Any]:
    graph = build_langgraph()
    return graph.invoke(create_initial_state(user_query="AI"))


def test_graph_executes_with_tracing_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "0")

    result = _invoke_clarification_graph()

    assert result["routing_decision"] == "clarification_node"
    assert result["workflow_status"] == "awaiting_clarification"
    assert result["retry_counts"]["blog_writer"] == 0


def test_graph_executes_with_tracing_enabled_mocked_tracer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "0")

    baseline = _invoke_clarification_graph()

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)
    try:
        traced_result = _invoke_clarification_graph()
    finally:
        observability_module.reset_tracer_factory()

    assert traced_result["routing_decision"] == baseline["routing_decision"]
    assert traced_result["workflow_status"] == baseline["workflow_status"]
    assert traced_result["retry_counts"] == baseline["retry_counts"]
    assert traced_result["cost_controls"] == baseline["cost_controls"]

    event_names = [item["event"] for item in recording.events]
    assert "workflow_start" in event_names
    assert "workflow_finish" in event_names
    assert "node_start" in event_names
    assert "node_finish" in event_names

    node_names = {
        str(item.get("node_name", "")).strip()
        for item in recording.events
        if item.get("event") == "node_start"
    }
    assert node_names
    assert node_names.issubset(AUTHORITATIVE_NODE_SET)
    assert {"query_handler_node", "clarification_node"}.issubset(node_names)

    node_finish_events = [
        item for item in recording.events if item.get("event") == "node_finish"
    ]
    assert node_finish_events
    for event in node_finish_events:
        metadata = event.get("metadata", {})
        assert isinstance(metadata, dict)
        assert isinstance(metadata.get("duration_ms"), int)
        assert int(metadata.get("duration_ms", -1)) >= 0
        assert metadata.get("node_started_at")
        assert metadata.get("node_ended_at")


def test_tracing_failure_does_not_fail_graph_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "0")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")

    observability_module.set_tracer_factory(lambda _config: _FailingTracer())
    try:
        result = _invoke_clarification_graph()
    finally:
        observability_module.reset_tracer_factory()

    assert result["routing_decision"] == "clarification_node"
    assert result["workflow_status"] == "awaiting_clarification"
    assert result["retry_counts"]["blog_writer"] == 0
