from __future__ import annotations

from contentblitz.ui.progress import create_progress_event
from contentblitz.ui.status import (
    build_initial_node_statuses,
    build_status_messages,
    derive_node_statuses,
    summarize_workflow_status,
)


def test_initial_statuses_are_pending_for_all_nodes() -> None:
    statuses = build_initial_node_statuses()
    assert len(statuses) == 12
    assert all(status == "pending" for status in statuses.values())


def test_derive_node_statuses_maps_running_completed_and_skipped() -> None:
    events = [
        create_progress_event(node_name="query_handler_node", status="running"),
        create_progress_event(node_name="query_handler_node", status="completed"),
        create_progress_event(node_name="retry_router_node", status="skipped"),
    ]
    statuses = derive_node_statuses(events)
    assert statuses["query_handler_node"] == "completed"
    assert statuses["retry_router_node"] == "skipped"


def test_degraded_research_message_is_user_safe() -> None:
    statuses = build_initial_node_statuses()
    statuses["research_agent_node"] = "degraded"
    state = {"workflow_status": "partial_success", "research_data": {"degraded": True}}
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert any("Research results are degraded" in message for message in messages)


def test_recoverable_image_failure_is_reported_as_warning() -> None:
    statuses = build_initial_node_statuses()
    statuses["image_agent_node"] = "degraded"
    state = {
        "workflow_status": "partial_success",
        "image_outputs": [{"status": "failed"}],
        "errors": [{"agent": "image_agent", "recoverable": True, "message": "safe warning"}],
    }
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert any("Image generation encountered a recoverable issue" in message for message in messages)


def test_export_failure_is_non_blocking_when_final_response_exists() -> None:
    statuses = build_initial_node_statuses()
    statuses["export_node"] = "degraded"
    state = {
        "workflow_status": "partial_success",
        "final_response": "Final text exists.",
        "export_metadata": {"error_log": [{"message": "pdf export failed"}]},
    }
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert any("exports failed" in message.lower() for message in messages)


def test_terminal_failure_message_is_safe() -> None:
    statuses = build_initial_node_statuses()
    statuses["error_handler_node"] = "failed"
    state = {"workflow_status": "failed", "errors": [{"message": "raw internal details"}]}
    summary = summarize_workflow_status(statuses, workflow_status=state["workflow_status"])
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert summary == "failed"
    assert any("terminal failure" in message.lower() for message in messages)
    assert all("Traceback" not in message for message in messages)
