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
