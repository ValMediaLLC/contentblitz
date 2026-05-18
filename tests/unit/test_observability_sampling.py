from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pytest

from contentblitz.core import observability as observability_module


def _clear_langsmith_and_sampling_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
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


def _trace_workflow(
    *,
    session_id: str,
    finish_metadata: Mapping[str, Any],
) -> _RecordingTracer:
    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)
    tracer = observability_module.get_workflow_tracer()
    workflow_span = tracer.start_workflow(
        metadata={
            "session_id": session_id,
            "requested_outputs": ["blog"],
            "workflow_status": "running",
        }
    )
    workflow_span.finish(
        metadata=finish_metadata,
        outputs={"workflow_status": str(finish_metadata.get("workflow_status", ""))},
    )
    return recording


def test_invalid_sampling_values_fall_back_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_and_sampling_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_SAMPLE_RATE", "not-a-float")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE", "9.0")

    config = observability_module.build_observability_config()

    assert config.trace_sample_rate == 1.0
    assert config.trace_failure_sample_rate == 1.0


def test_sample_rate_one_traces_success_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_and_sampling_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_SAMPLE_RATE", "1.0")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE", "1.0")

    recording = _trace_workflow(
        session_id="session-trace-all",
        finish_metadata={"workflow_status": "success"},
    )

    events = [event["event"] for event in recording.events]
    assert "workflow_start" in events
    assert "workflow_finish" in events


def test_sample_rate_zero_skips_success_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_and_sampling_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_SAMPLE_RATE", "0.0")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE", "0.0")

    recording = _trace_workflow(
        session_id="session-skip-success",
        finish_metadata={"workflow_status": "success"},
    )

    assert recording.events == []


def test_failure_sample_rate_can_trace_failed_workflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_and_sampling_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_SAMPLE_RATE", "0.0")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE", "1.0")

    success_recording = _trace_workflow(
        session_id="session-failure-sample",
        finish_metadata={"workflow_status": "success"},
    )
    failed_recording = _trace_workflow(
        session_id="session-failure-sample",
        finish_metadata={"workflow_status": "failed"},
    )

    assert success_recording.events == []
    assert [event["event"] for event in failed_recording.events] == [
        "workflow_start",
        "workflow_finish",
    ]


def test_sampling_decision_is_deterministic_for_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_and_sampling_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_SAMPLE_RATE", "0.42")
    monkeypatch.setenv("CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE", "0.84")
    config = observability_module.build_observability_config()

    decision_a = observability_module._build_sampling_decision(  # noqa: SLF001
        metadata={"session_id": "session-deterministic"},
        config=config,
    )
    decision_b = observability_module._build_sampling_decision(  # noqa: SLF001
        metadata={"session_id": "session-deterministic"},
        config=config,
    )

    assert decision_a == decision_b
