from __future__ import annotations

from datetime import UTC, datetime, timedelta

from contentblitz.core import observability as observability_module


def test_node_timing_metadata_uses_total_node_duration_from_timestamps() -> None:
    started_at = datetime(2026, 5, 20, 13, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=1250)

    metadata = observability_module.build_node_timing_metadata(
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=None,
    )

    assert metadata["duration_ms"] == 1250
    assert metadata["node_started_at"].startswith("2026-05-20T13:00:00")
    assert metadata["node_ended_at"].startswith("2026-05-20T13:00:01.250")


def test_node_timing_metadata_preserves_explicit_duration_value() -> None:
    started_at = datetime(2026, 5, 20, 13, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=900)

    metadata = observability_module.build_node_timing_metadata(
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=300,
    )

    assert metadata["duration_ms"] == 300


def test_research_node_metadata_uses_provider_specific_latency_fields() -> None:
    started_at = datetime(2026, 5, 20, 13, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=1200)
    state = {
        "workflow_status": "running",
        "research_data": {},
    }
    updates = {
        "workflow_status": "research_complete",
        "research_data": {
            "cache_hit": False,
            "provider_latency_total_ms": 1800,
            "provider_latency_wall_ms": 900,
            "provider_latency_by_provider_ms": {"serp_api": 1200, "openai": 600},
            "provider_call_count": 4,
            "provider_call_count_by_provider": {"serp_api": 3, "openai": 1},
            "provider_timeout_count": 1,
            "provider_timeout_count_by_provider": {"serp_api": 1},
            "search_provider_wall_timeout_ms": 8000,
            "search_provider_wall_timeout_triggered": True,
        },
        "sources": [{"provider": "serp_api", "title": "Source", "snippet": "safe"}],
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="research_agent_node",
        node_status="completed",
        updates=updates,
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=1200,
    )

    assert metadata["duration_ms"] == 1200
    assert metadata["provider_latency_total_ms"] == 1800
    assert metadata["provider_latency_wall_ms"] == 900
    assert metadata["provider_latency_by_provider_ms"]["serp_api"] == 1200
    assert metadata["provider_timeout_count_by_provider"]["serp_api"] == 1
    assert metadata["search_provider_wall_timeout_triggered"] is True
    assert "provider_latency_ms" not in metadata


def test_content_strategist_metadata_uses_async_latency_fields() -> None:
    started_at = datetime(2026, 5, 20, 13, 0, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(milliseconds=1200)
    state = {"workflow_status": "running", "tool_outputs": {}}
    updates = {
        "workflow_status": "strategy_complete",
        "tool_outputs": {
            "content_strategist": {
                "provider": "openai",
                "model": "gpt-4o",
                "provider_call_count": 3,
                "provider_latency_ms": 700,
                "provider_latency_total_ms": 1800,
                "provider_latency_wall_ms": 700,
                "provider_latency_by_output_type_ms": {
                    "blog": 700,
                    "linkedin": 600,
                    "image": 500,
                },
                "provider_call_count_by_output_type": {
                    "blog": 1,
                    "linkedin": 1,
                    "image": 1,
                },
            }
        },
    }

    metadata = observability_module.safe_node_end_metadata(
        state=state,
        node_name="content_strategist_node",
        node_status="completed",
        updates=updates,
        node_started_at=started_at,
        node_ended_at=ended_at,
        duration_ms=1200,
    )

    assert metadata["duration_ms"] == 1200
    assert metadata["provider"] == "openai"
    assert metadata["model"] == "gpt-4o"
    assert metadata["provider_call_count"] == 3
    assert metadata["provider_latency_ms"] == 700
    assert metadata["provider_latency_total_ms"] == 1800
    assert metadata["provider_latency_wall_ms"] == 700
    assert metadata["provider_latency_by_output_type_ms"]["blog"] == 700
    assert metadata["provider_call_count_by_output_type"]["image"] == 1
