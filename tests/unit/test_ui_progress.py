from __future__ import annotations

import pytest

from contentblitz.ui.progress import (
    VALID_PROGRESS_STATUSES,
    build_pending_progress_events,
    create_progress_event,
    normalize_progress_status,
    order_progress_events,
)
from contentblitz.workflow.routing import AUTHORITATIVE_NODES


def test_all_12_nodes_produce_valid_pending_events() -> None:
    events = build_pending_progress_events(timestamp="2026-05-10T10:00:00+00:00")
    assert len(events) == 12
    assert [event.node_name for event in events] == AUTHORITATIVE_NODES
    assert all(event.status == "pending" for event in events)


def test_unknown_node_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_progress_event(node_name="unknown_node", status="running")


def test_running_completed_and_skipped_statuses_are_preserved() -> None:
    running = create_progress_event(
        node_name="query_handler_node",
        status="running",
        timestamp="2026-05-10T10:00:00+00:00",
    )
    completed = create_progress_event(
        node_name="query_handler_node",
        status="completed",
        timestamp="2026-05-10T10:00:01+00:00",
    )
    skipped = create_progress_event(
        node_name="retry_router_node",
        status="skipped",
        timestamp="2026-05-10T10:00:02+00:00",
    )
    assert running.status == "running"
    assert completed.status == "completed"
    assert skipped.status == "skipped"
    assert all(item.status in VALID_PROGRESS_STATUSES for item in (running, completed, skipped))


def test_invalid_status_safely_normalizes() -> None:
    assert normalize_progress_status("not-a-real-status") == "degraded"
    assert normalize_progress_status("invalid", invalid_fallback="failed") == "failed"


def test_progress_event_ordering_is_deterministic() -> None:
    events = [
        create_progress_event(
            node_name="blog_writer_node",
            status="completed",
            timestamp="2026-05-10T10:00:05+00:00",
        ),
        create_progress_event(
            node_name="query_handler_node",
            status="running",
            timestamp="2026-05-10T10:00:01+00:00",
        ),
        create_progress_event(
            node_name="query_handler_node",
            status="completed",
            timestamp="2026-05-10T10:00:02+00:00",
        ),
        {
            "node_name": "blog_writer_node",
            "status": "running",
            "timestamp": "2026-05-10T10:00:04+00:00",
            "message": "blog running",
        },
    ]

    ordered_once = order_progress_events(events)
    ordered_twice = order_progress_events(events)
    assert [(event.node_name, event.status, event.timestamp) for event in ordered_once] == [
        (event.node_name, event.status, event.timestamp) for event in ordered_twice
    ]
    assert ordered_once[0].node_name == "query_handler_node"
    assert ordered_once[0].status == "running"
