from __future__ import annotations

from contentblitz.ui.progress import create_progress_event
from contentblitz.ui.status import (
    apply_optional_node_skips,
    build_initial_node_statuses,
    build_status_messages,
    derive_node_statuses,
    summarize_workflow_status,
    workflow_requires_clarification,
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


def test_degraded_research_blog_flow_aggregates_to_partial_success() -> None:
    statuses = build_initial_node_statuses()
    statuses["research_agent_node"] = "degraded"
    statuses["blog_writer_node"] = "completed"
    statuses["output_assembler_node"] = "completed"
    state = {
        "workflow_status": "success",
        "research_data": {"degraded": True},
        "requested_outputs": ["blog", "research"],
    }
    summary = summarize_workflow_status(
        statuses, workflow_status=state["workflow_status"]
    )
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert summary == "partial_success"
    assert any("Research results are degraded" in message for message in messages)


def test_degraded_research_linkedin_flow_aggregates_to_partial_success() -> None:
    statuses = build_initial_node_statuses()
    statuses["research_agent_node"] = "degraded"
    statuses["linkedin_writer_node"] = "completed"
    statuses["output_assembler_node"] = "completed"
    state = {
        "workflow_status": "success",
        "research_data": {"degraded": True},
        "requested_outputs": ["linkedin", "research"],
    }
    summary = summarize_workflow_status(
        statuses, workflow_status=state["workflow_status"]
    )
    assert summary == "partial_success"


def test_degraded_image_only_flow_aggregates_to_partial_success() -> None:
    statuses = build_initial_node_statuses()
    statuses["image_agent_node"] = "degraded"
    statuses["output_assembler_node"] = "completed"
    state = {
        "workflow_status": "success",
        "requested_outputs": ["image"],
        "image_outputs": [{"status": "failed"}],
    }
    summary = summarize_workflow_status(
        statuses, workflow_status=state["workflow_status"]
    )
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert summary == "partial_success"
    assert any("recoverable issue" in message for message in messages)


def test_clarification_flow_aggregates_to_awaiting_clarification() -> None:
    statuses = build_initial_node_statuses()
    statuses["query_handler_node"] = "completed"
    statuses["clarification_node"] = "completed"
    state = {
        "workflow_status": "awaiting_clarification",
        "clarification_needed": True,
        "requested_outputs": [],
    }
    clarification_required = workflow_requires_clarification(
        state=state,
        node_statuses=statuses,
    )
    summary = summarize_workflow_status(
        statuses,
        workflow_status=state["workflow_status"],
        clarification_required=clarification_required,
    )
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert summary == "awaiting_clarification"
    assert not any("completed successfully" in message.lower() for message in messages)
    assert any("awaiting clarification" in message.lower() for message in messages)


def test_clean_flow_aggregates_to_success() -> None:
    statuses = build_initial_node_statuses()
    statuses["query_handler_node"] = "completed"
    statuses["content_strategist_node"] = "completed"
    statuses["blog_writer_node"] = "completed"
    statuses["quality_validator_node"] = "completed"
    statuses["output_assembler_node"] = "completed"
    state = {"workflow_status": "success", "requested_outputs": ["blog"]}
    summary = summarize_workflow_status(
        statuses, workflow_status=state["workflow_status"]
    )
    assert summary == "success"


def test_recoverable_image_failure_is_reported_as_warning() -> None:
    statuses = build_initial_node_statuses()
    statuses["image_agent_node"] = "degraded"
    state = {
        "workflow_status": "partial_success",
        "image_outputs": [{"status": "failed"}],
        "errors": [
            {"agent": "image_agent", "recoverable": True, "message": "safe warning"}
        ],
        "requested_outputs": ["image"],
    }
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert any(
        "Image generation encountered a recoverable issue" in message
        for message in messages
    )


def test_export_failure_is_non_blocking_when_final_response_exists() -> None:
    statuses = build_initial_node_statuses()
    statuses["export_node"] = "degraded"
    state = {
        "workflow_status": "partial_success",
        "final_response": "Final text exists.",
        "export_requested": True,
        "export_metadata": {
            "formats_requested": ["pdf"],
            "error_log": [{"message": "pdf export failed"}],
            "failed_export_formats": ["pdf"],
            "export_error_count": 1,
        },
    }
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert any("exports failed" in message.lower() for message in messages)


def test_export_warning_without_failed_formats_does_not_emit_failure_message() -> None:
    statuses = build_initial_node_statuses()
    statuses["export_node"] = "completed"
    state = {
        "workflow_status": "success",
        "final_response": "Final text exists.",
        "export_requested": True,
        "export_metadata": {
            "formats_requested": ["pdf"],
            "export_status": {"pdf": "completed"},
            "error_log": [
                {"code": "pdf_validation_warning", "message": "safe warning"}
            ],
            "failed_export_formats": [],
            "export_error_count": 0,
            "export_warning_count": 1,
        },
    }

    messages = build_status_messages(state=state, node_statuses=statuses)

    assert not any("exports failed" in message.lower() for message in messages)
    assert any("non-blocking warnings" in message.lower() for message in messages)


def test_export_off_marks_export_node_skipped() -> None:
    statuses = build_initial_node_statuses()
    statuses["export_node"] = "completed"
    state = {
        "workflow_status": "success",
        "export_requested": False,
        "export_metadata": {"formats_requested": []},
    }
    normalized = apply_optional_node_skips(state=state, node_statuses=statuses)
    assert normalized["export_node"] == "skipped"


def test_terminal_failure_message_is_safe() -> None:
    statuses = build_initial_node_statuses()
    statuses["error_handler_node"] = "failed"
    state = {
        "workflow_status": "failed",
        "errors": [{"message": "raw internal details"}],
    }
    summary = summarize_workflow_status(
        statuses, workflow_status=state["workflow_status"]
    )
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert summary == "failed"
    assert any("terminal failure" in message.lower() for message in messages)
    assert all("Traceback" not in message for message in messages)


def test_status_messages_exclude_none_and_empty_entries() -> None:
    statuses = build_initial_node_statuses()
    statuses["output_assembler_node"] = "completed"
    state = {
        "workflow_status": "success",
        "requested_outputs": ["blog"],
        "errors": [{"message": None, "recoverable": True}],
        "export_metadata": {"formats_requested": [], "error_log": []},
    }
    messages = build_status_messages(state=state, node_statuses=statuses)
    assert all(message.strip() for message in messages)
    assert all(message.lower() not in {"none", "null"} for message in messages)
