from __future__ import annotations

from datetime import UTC, datetime, timedelta

from contentblitz.core import observability as observability_module


def test_provider_latency_is_not_derived_from_node_duration() -> None:
    started_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=180)
    state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
        "content_drafts": {"blog": {"model_used": "gpt-4o"}},
    }
    updates = {
        "workflow_status": "success",
        "content_drafts": {"blog": {"model_used": "gpt-4o"}},
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="blog_writer_node",
        node_status="completed",
        updates=updates,
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=180,
    )

    assert metadata["duration_ms"] == 180
    assert metadata["provider"] == "openai"
    assert metadata["model"] == "gpt-4o"
    assert "provider_latency_ms" not in metadata


def test_provider_latency_is_emitted_when_explicitly_measured() -> None:
    started_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=180)
    state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
        "content_drafts": {
            "blog": {
                "model_used": "gpt-4o",
                "provider_latency_ms": 42,
                "provider_call_count": 2,
            }
        },
    }
    updates = {
        "workflow_status": "success",
        "content_drafts": {
            "blog": {
                "model_used": "gpt-4o",
                "provider_latency_ms": 42,
                "provider_call_count": 2,
            }
        },
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="blog_writer_node",
        node_status="completed",
        updates=updates,
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=180,
    )

    assert metadata["duration_ms"] == 180
    assert metadata["provider_latency_ms"] == 42
    assert metadata["provider_call_count"] == 2
    assert metadata["provider_latency_ms"] != metadata["duration_ms"]


def test_writer_provider_infers_anthropic_from_claude_model() -> None:
    started_at = datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=120)
    state = {
        "workflow_status": "running",
        "requested_outputs": ["blog"],
        "content_drafts": {"blog": {"model_used": "claude-sonnet-4-6"}},
    }
    updates = {
        "workflow_status": "success",
        "content_drafts": {"blog": {"model_used": "claude-sonnet-4-6"}},
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="blog_writer_node",
        node_status="completed",
        updates=updates,
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=120,
    )

    assert metadata["provider"] == "anthropic"
    assert metadata["model"] == "claude-sonnet-4-6"
