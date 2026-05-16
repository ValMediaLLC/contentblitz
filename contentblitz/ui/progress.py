"""UI-safe workflow progress event helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from contentblitz.workflow.routing import AUTHORITATIVE_NODE_SET, AUTHORITATIVE_NODES

VALID_PROGRESS_STATUSES = (
    "pending",
    "running",
    "completed",
    "skipped",
    "degraded",
    "failed",
)
_VALID_PROGRESS_STATUS_SET = set(VALID_PROGRESS_STATUSES)
_INVALID_STATUS_FALLBACKS = {"degraded", "failed"}
_NODE_ORDER = {name: idx for idx, name in enumerate(AUTHORITATIVE_NODES)}
_STATUS_ORDER = {
    "pending": 0,
    "running": 1,
    "completed": 2,
    "skipped": 3,
    "degraded": 4,
    "failed": 5,
}


@dataclass(frozen=True)
class UIProgressEvent:
    """Normalized progress event exposed to the frontend rendering layer."""

    node_name: str
    status: str
    message: str
    timestamp: str
    safe_metadata: dict[str, Any] = field(default_factory=dict)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_timestamp(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _utc_now_iso()


def _default_message(node_name: str, status: str) -> str:
    status_text = status.replace("_", " ").strip()
    return f"{node_name}: {status_text}."


def _sanitize_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _sanitize_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata_value(item) for item in value]
    return str(value)


def _sanitize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    return {
        str(key): _sanitize_metadata_value(value) for key, value in metadata.items()
    }


def validate_node_name(node_name: str) -> str:
    """Return a validated authoritative node name or raise ValueError."""
    normalized = str(node_name).strip()
    if normalized not in AUTHORITATIVE_NODE_SET:
        raise ValueError(f"Unknown workflow node: {normalized}")
    return normalized


def normalize_progress_status(
    status: str,
    *,
    invalid_fallback: str = "degraded",
) -> str:
    """Normalize unknown statuses to degraded/failed in a deterministic way."""
    normalized = str(status).strip().lower()
    if normalized in _VALID_PROGRESS_STATUS_SET:
        return normalized

    fallback = str(invalid_fallback).strip().lower()
    if fallback not in _INVALID_STATUS_FALLBACKS:
        fallback = "degraded"
    return fallback


def create_progress_event(
    *,
    node_name: str,
    status: str,
    message: str = "",
    timestamp: str | None = None,
    safe_metadata: Mapping[str, Any] | None = None,
    invalid_status_fallback: str = "degraded",
) -> UIProgressEvent:
    """Create a validated, normalized progress event for a workflow node."""
    validated_node = validate_node_name(node_name)
    normalized_status = normalize_progress_status(
        status,
        invalid_fallback=invalid_status_fallback,
    )
    safe_message = str(message).strip() or _default_message(
        validated_node, normalized_status
    )
    return UIProgressEvent(
        node_name=validated_node,
        status=normalized_status,
        message=safe_message,
        timestamp=_safe_timestamp(timestamp),
        safe_metadata=_sanitize_metadata(safe_metadata),
    )


def build_pending_progress_events(
    *,
    node_names: Iterable[str] | None = None,
    timestamp: str | None = None,
) -> list[UIProgressEvent]:
    """Create a deterministic pending event list for the authoritative nodes."""
    names = list(node_names) if node_names is not None else list(AUTHORITATIVE_NODES)
    events: list[UIProgressEvent] = []
    seen: set[str] = set()
    for raw_name in names:
        validated = validate_node_name(str(raw_name))
        if validated in seen:
            continue
        seen.add(validated)
        events.append(
            create_progress_event(
                node_name=validated,
                status="pending",
                timestamp=timestamp,
            )
        )
    return events


def _timestamp_sort_key(timestamp: str) -> tuple[int, str]:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        utc_value = parsed.astimezone(timezone.utc).isoformat(timespec="seconds")
        return (0, utc_value)
    except ValueError:
        return (1, timestamp)


def _coerce_event(event: UIProgressEvent | Mapping[str, Any]) -> UIProgressEvent | None:
    if isinstance(event, UIProgressEvent):
        return event
    if not isinstance(event, Mapping):
        return None

    node_name = str(event.get("node_name", "")).strip()
    if node_name not in AUTHORITATIVE_NODE_SET:
        return None

    return create_progress_event(
        node_name=node_name,
        status=str(event.get("status", "degraded")),
        message=str(event.get("message", "")),
        timestamp=str(event.get("timestamp", "")).strip() or None,
        safe_metadata=(
            event.get("safe_metadata")
            if isinstance(event.get("safe_metadata"), Mapping)
            else None
        ),
    )


def order_progress_events(
    events: Iterable[UIProgressEvent | Mapping[str, Any]],
) -> list[UIProgressEvent]:
    """Return progress events in deterministic order."""
    collected: list[tuple[int, UIProgressEvent]] = []
    for idx, raw in enumerate(events):
        event = _coerce_event(raw)
        if event is None:
            continue
        collected.append((idx, event))

    return [
        event
        for _, event in sorted(
            collected,
            key=lambda item: (
                _timestamp_sort_key(item[1].timestamp),
                _NODE_ORDER.get(item[1].node_name, len(_NODE_ORDER)),
                _STATUS_ORDER.get(item[1].status, len(_STATUS_ORDER)),
                item[0],
            ),
        )
    ]
