from __future__ import annotations

from typing import Any

import frontend.services.orchestrator_client as orchestrator_client_module
from frontend.services.orchestrator_client import stream_workflow_progress


def test_streaming_node_timing_stays_total_and_provider_latency_is_explicit_only(
    monkeypatch,
) -> None:
    class _FakeGraph:
        def stream(self, _state, *, stream_mode):
            assert stream_mode == ["tasks", "updates", "values"]
            yield ("values", {"workflow_status": "running"})
            yield (
                "tasks",
                {
                    "id": "task-1",
                    "name": "query_handler_node",
                    "input": {},
                    "triggers": ["start"],
                },
            )
            yield (
                "updates",
                {"query_handler_node": {"workflow_status": "routing_complete"}},
            )
            yield (
                "tasks",
                {
                    "id": "task-2",
                    "name": "blog_writer_node",
                    "input": {},
                    "triggers": ["start"],
                },
            )
            yield (
                "updates",
                {
                    "blog_writer_node": {
                        "workflow_status": "writing_complete",
                        "content_drafts": {
                            "blog": {
                                "body": "safe blog body",
                                "model_used": "gpt-4o",
                            }
                        },
                    }
                },
            )
            yield ("values", {"workflow_status": "partial_success"})

    monkeypatch.setattr(orchestrator_client_module, "_get_graph", lambda: _FakeGraph())

    progress_events: list[dict[str, Any]] = []
    final_result: dict[str, Any] = {}
    for item in stream_workflow_progress(
        user_query="write a blog",
        requested_outputs=["blog"],
        export_requested=False,
        export_formats=[],
    ):
        if item.get("type") == "progress":
            event = item.get("event")
            if isinstance(event, dict):
                progress_events.append(event)
        elif item.get("type") == "final":
            payload = item.get("result")
            if isinstance(payload, dict):
                final_result = payload

    assert final_result.get("workflow_status") == "partial_success"

    completed_events = [
        event
        for event in progress_events
        if event.get("status") in {"completed", "degraded", "failed", "skipped"}
    ]
    assert completed_events

    blog_event = next(
        event
        for event in completed_events
        if event.get("node_name") == "blog_writer_node"
    )
    safe_metadata = blog_event.get("safe_metadata", {})
    assert isinstance(safe_metadata.get("duration_ms"), int)
    assert safe_metadata["duration_ms"] >= 0
    assert safe_metadata.get("provider") == "openai"
    assert safe_metadata.get("model") == "gpt-4o"
    assert "provider_latency_ms" not in safe_metadata

    query_event = next(
        event
        for event in completed_events
        if event.get("node_name") == "query_handler_node"
    )
    query_metadata = query_event.get("safe_metadata", {})
    assert "provider_latency_ms" not in query_metadata
