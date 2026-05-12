"""Safe serialization helpers for persisted workflow runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping
from uuid import uuid4

from contentblitz.persistence.models import PersistedRunRecord, PersistedRunSummary
from contentblitz.ui.error_display import normalize_errors_for_display, redact_sensitive_text
from contentblitz.ui.rendering import dedupe_sources_for_display, sanitize_image_outputs_for_display

_TRACEBACK_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
)
_SAFE_SIGNAL_RE = re.compile(r"^[a-z0-9_]{1,64}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _sanitize_text(value: Any) -> str:
    cleaned = _safe_text(value)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in {"none", "null"}:
        return ""
    if any(marker in lowered for marker in _TRACEBACK_MARKERS):
        return "Internal details were removed."
    return redact_sensitive_text(cleaned)


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sanitize_requested_outputs(outputs: Any) -> List[str]:
    allowed = {"research", "blog", "linkedin", "image"}
    normalized: List[str] = []
    for item in _safe_list(outputs):
        token = _safe_text(item).lower()
        if token in allowed and token not in normalized:
            normalized.append(token)
    return normalized


def _sanitize_content_drafts(content_drafts: Any) -> Dict[str, Any]:
    drafts = _safe_dict(content_drafts)
    blog = _safe_dict(drafts.get("blog", {}))
    linkedin = _safe_dict(drafts.get("linkedin", {}))
    research = _safe_dict(drafts.get("research_report", {}))
    return {
        "blog": {
            "body": _sanitize_text(blog.get("body")),
            "version": int(blog.get("version", 0)) if isinstance(blog.get("version"), int) else 0,
        },
        "linkedin": {
            "body": _sanitize_text(linkedin.get("body")),
            "version": int(linkedin.get("version", 0))
            if isinstance(linkedin.get("version"), int)
            else 0,
        },
        "research_report": {
            "body": _sanitize_text(research.get("body")),
        },
    }


def _sanitize_progress_events(events: Any) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for item in _safe_list(events):
        if not isinstance(item, Mapping):
            continue
        node_name = _safe_text(item.get("node_name"))
        status = _safe_text(item.get("status")).lower()
        timestamp = _safe_text(item.get("timestamp"))
        message = _sanitize_text(item.get("message"))
        safe_metadata = _safe_dict(item.get("safe_metadata"))
        safe_metadata.pop("workflow_status", None)
        safe_metadata.pop("ui_workflow_status", None)
        if not node_name or not status:
            continue
        sanitized.append(
            {
                "node_name": node_name,
                "status": status,
                "timestamp": timestamp,
                "message": message,
                "safe_metadata": deepcopy(safe_metadata),
            }
        )
    return sanitized


def _sanitize_quality_scores(quality_scores: Any) -> Dict[str, Any]:
    raw = _safe_dict(quality_scores)
    cleaned: Dict[str, Any] = {}
    for key in ("blog", "linkedin", "image"):
        score = _safe_dict(raw.get(key, {}))
        summary: Dict[str, Any] = {}
        status = _safe_text(score.get("validation_status")).lower()
        if status:
            summary["validation_status"] = status
        composite = score.get("composite")
        if isinstance(composite, (int, float)):
            summary["composite"] = float(composite)
        if summary:
            cleaned[key] = summary
    citation = _safe_dict(raw.get("citation_validation", {}))
    citation_summary: Dict[str, Any] = {}
    citation_status = _safe_text(citation.get("status")).lower()
    if citation_status:
        citation_summary["status"] = citation_status
    for field_name in (
        "invalid_count",
        "duplicate_count",
        "unsafe_url_count",
        "missing_count",
        "valid_source_count",
    ):
        value = citation.get(field_name)
        if isinstance(value, int) and value >= 0:
            citation_summary[field_name] = value
    if citation_summary:
        cleaned["citation_validation"] = citation_summary
    return cleaned


def _sanitize_export_metadata(export_metadata: Any) -> Dict[str, Any]:
    meta = _safe_dict(export_metadata)
    formats = [token for token in [_safe_text(item).lower() for item in _safe_list(meta.get("formats_requested", []))] if token]
    raw_paths = _safe_dict(meta.get("export_paths", {}))
    raw_status = _safe_dict(meta.get("export_status", {}))
    paths: Dict[str, str] = {}
    status: Dict[str, str] = {}
    for key, value in raw_paths.items():
        fmt = _safe_text(key).lower()
        path_value = _safe_text(value)
        if not fmt or not path_value:
            continue
        paths[fmt] = path_value
    for key, value in raw_status.items():
        fmt = _safe_text(key).lower()
        status_value = _safe_text(value).lower()
        if not fmt or not status_value:
            continue
        status[fmt] = status_value
    raw_error_count = meta.get("export_error_count", 0)
    export_error_count = raw_error_count if isinstance(raw_error_count, int) and raw_error_count >= 0 else 0
    status_messages = [
        _sanitize_text(item)
        for item in _safe_list(meta.get("status_messages", []))
        if _sanitize_text(item)
    ]
    return {
        "formats_requested": formats,
        "export_paths": paths,
        "export_status": status,
        "export_error_count": export_error_count,
        "status_messages": status_messages,
    }


def _sanitize_sources(sources: Any) -> List[Dict[str, Any]]:
    deduped = dedupe_sources_for_display(_safe_list(sources))
    sanitized: List[Dict[str, Any]] = []
    for item in deduped:
        if not isinstance(item, Mapping):
            continue
        sanitized.append(
            {
                "title": _sanitize_text(item.get("title")),
                "url": _safe_text(item.get("url")) or None,
                "snippet": _sanitize_text(item.get("snippet")),
                "source": _sanitize_text(item.get("source")),
                "published_at": _safe_text(item.get("published_at")) or None,
                "citation_available": bool(item.get("citation_available", False)),
                "credibility_score": float(item.get("credibility_score", 0.0))
                if isinstance(item.get("credibility_score"), (int, float))
                else 0.0,
            }
        )
    return sanitized


def _sanitize_partial_outputs(partial_outputs: Any) -> Dict[str, str]:
    raw = _safe_dict(partial_outputs)
    return {
        "blog": _sanitize_text(raw.get("blog")),
        "linkedin": _sanitize_text(raw.get("linkedin")),
        "research": _sanitize_text(raw.get("research")),
    }


def _sanitize_status_messages(messages: Any) -> List[str]:
    cleaned: List[str] = []
    for item in _safe_list(messages):
        text = _sanitize_text(item)
        if not text:
            continue
        cleaned.append(text)
    return cleaned


def _sanitize_injection_signals(signals: Any) -> List[str]:
    cleaned: List[str] = []
    for item in _safe_list(signals):
        token = _safe_text(item).lower()
        if not token:
            continue
        if not _SAFE_SIGNAL_RE.match(token):
            continue
        if token not in cleaned:
            cleaned.append(token)
    return cleaned


def _sanitize_warnings(warnings: Any) -> List[str]:
    return _sanitize_status_messages(warnings)


def _sanitize_errors(errors: Any) -> List[Dict[str, Any]]:
    normalized = normalize_errors_for_display(_safe_list(errors))
    cleaned: List[Dict[str, Any]] = []
    for item in normalized:
        if not isinstance(item, Mapping):
            continue
        cleaned_item = {
            "code": _sanitize_text(item.get("code")) or "unknown_error",
            "message": _sanitize_text(item.get("message")),
            "recoverable": bool(item.get("recoverable", False)),
        }
        agent = _sanitize_text(item.get("agent"))
        if agent:
            cleaned_item["agent"] = agent
        provider = _sanitize_text(item.get("provider"))
        if provider:
            cleaned_item["provider"] = provider
        cleaned.append(cleaned_item)
    return cleaned


def _sanitize_image_outputs(image_outputs: Any) -> List[Dict[str, Any]]:
    def _normalize_image_error(error_value: Any) -> Dict[str, Any]:
        raw = _safe_dict(error_value)
        recoverable = bool(raw.get("recoverable", True))
        return {
            "code": "image_generation_failed",
            "message": (
                "Image generation encountered a recoverable issue."
                if recoverable
                else "Image generation failed."
            ),
            "recoverable": recoverable,
        }

    safe = sanitize_image_outputs_for_display(_safe_list(image_outputs))
    cleaned: List[Dict[str, Any]] = []
    for item in safe:
        if not isinstance(item, Mapping):
            continue
        raw = dict(item)
        raw.pop("base64", None)
        raw.pop("b64_json", None)
        if isinstance(raw.get("url"), str) and raw["url"].strip().lower().startswith("data:image/"):
            continue
        if isinstance(raw.get("error"), Mapping):
            raw["error"] = _normalize_image_error(raw.get("error"))
        cleaned.append(raw)
    return cleaned


def _sanitize_ui_selected_options(options: Any) -> Dict[str, Any]:
    raw = _safe_dict(options)
    outputs = _sanitize_requested_outputs(raw.get("requested_outputs", []))
    export_requested = bool(raw.get("export_requested", False))
    export_formats = [token for token in [_safe_text(item).lower() for item in _safe_list(raw.get("export_formats", []))] if token]
    return {
        "requested_outputs": outputs,
        "export_requested": export_requested,
        "export_formats": export_formats,
    }


def _sanitize_node_statuses(statuses: Any) -> Dict[str, str]:
    raw = _safe_dict(statuses)
    cleaned: Dict[str, str] = {}
    for key, value in raw.items():
        node = _safe_text(key)
        status = _safe_text(value).lower()
        if not node or not status:
            continue
        cleaned[node] = status
    return cleaned


def _sanitize_export_paths_for_restore(export_paths: Mapping[str, Any]) -> tuple[Dict[str, str], List[str]]:
    safe_paths: Dict[str, str] = {}
    missing: List[str] = []
    for fmt, path_value in export_paths.items():
        fmt_name = _safe_text(fmt).lower()
        path_text = _safe_text(path_value)
        if not fmt_name or not path_text:
            continue
        try:
            if Path(path_text).exists():
                safe_paths[fmt_name] = path_text
            else:
                missing.append(fmt_name)
        except OSError:
            missing.append(fmt_name)
    return safe_paths, missing


def serialize_workflow_run(
    *,
    result_state: Mapping[str, Any],
    ui_selected_options: Mapping[str, Any] | None = None,
    progress_events: List[Mapping[str, Any]] | None = None,
    status_messages: List[str] | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    created_at: str | None = None,
) -> Dict[str, Any]:
    """Serialize a workflow run into a safe persistable payload."""
    state = deepcopy(dict(result_state))
    now = _utc_now_iso()
    persisted_run_id = _safe_text(run_id) or uuid4().hex
    persisted_session_id = _safe_text(session_id) or uuid4().hex
    created = _safe_text(created_at) or now
    updated = now

    requested_outputs = _sanitize_requested_outputs(state.get("requested_outputs", []))
    content_drafts = _sanitize_content_drafts(state.get("content_drafts", {}))
    image_prompts = [_sanitize_text(item) for item in _safe_list(state.get("image_prompts", [])) if _sanitize_text(item)]
    image_outputs = _sanitize_image_outputs(state.get("image_outputs", []))
    sources = _sanitize_sources(state.get("sources", []))
    quality_scores = _sanitize_quality_scores(state.get("quality_scores", {}))
    export_metadata = _sanitize_export_metadata(state.get("export_metadata", {}))
    errors = _sanitize_errors(state.get("errors", []))
    warnings = _sanitize_warnings(state.get("warnings", []))
    partial_outputs = _sanitize_partial_outputs(state.get("partial_outputs", {}))
    safe_progress_events = _sanitize_progress_events(progress_events or _safe_list(state.get("ui_progress_events", [])))
    safe_status_messages = _sanitize_status_messages(status_messages or _safe_list(state.get("status_messages", [])))
    ui_options = _sanitize_ui_selected_options(ui_selected_options or state.get("ui_selected_options", {}))
    ui_node_statuses = _sanitize_node_statuses(state.get("ui_node_statuses", {}))
    ui_workflow_status = _sanitize_text(state.get("ui_workflow_status") or state.get("workflow_status"))
    prompt_injection_detected = bool(state.get("prompt_injection_detected", False))
    prompt_injection_signals = _sanitize_injection_signals(
        state.get("prompt_injection_signals", [])
    )
    sanitized_user_query = _sanitize_text(state.get("sanitized_user_query"))

    record = PersistedRunRecord(
        run_id=persisted_run_id,
        session_id=persisted_session_id,
        created_at=created,
        updated_at=updated,
        user_query=_sanitize_text(state.get("user_query")),
        requested_outputs=requested_outputs,
        workflow_status=ui_workflow_status,
        routing_decision=_sanitize_text(state.get("routing_decision")),
        final_response=_sanitize_text(state.get("final_response")),
        content_drafts=content_drafts,
        partial_outputs=partial_outputs,
        partial_output_mode=_sanitize_text(state.get("partial_output_mode")) or "none",
        image_prompts=image_prompts,
        image_outputs=image_outputs,
        sources=sources,
        quality_scores=quality_scores,
        export_metadata=export_metadata,
        warnings=warnings,
        errors=errors,
        progress_events=safe_progress_events,
        status_messages=safe_status_messages,
        ui_selected_options=ui_options,
        ui_node_statuses=ui_node_statuses,
        ui_workflow_status=ui_workflow_status,
        prompt_injection_detected=prompt_injection_detected,
        prompt_injection_signals=prompt_injection_signals,
        sanitized_user_query=sanitized_user_query,
    )
    return asdict(record)


def deserialize_workflow_run(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a safe UI-restorable state object from persisted run data."""
    payload = deepcopy(dict(record))
    requested_outputs = _sanitize_requested_outputs(payload.get("requested_outputs", []))
    content_drafts = _sanitize_content_drafts(payload.get("content_drafts", {}))
    partial_outputs = _sanitize_partial_outputs(payload.get("partial_outputs", {}))
    image_outputs = _sanitize_image_outputs(payload.get("image_outputs", []))
    sources = _sanitize_sources(payload.get("sources", []))
    export_metadata = _sanitize_export_metadata(payload.get("export_metadata", {}))
    export_paths, missing_paths = _sanitize_export_paths_for_restore(export_metadata.get("export_paths", {}))
    export_metadata["export_paths"] = export_paths

    warnings = _sanitize_warnings(payload.get("warnings", []))
    if missing_paths:
        warnings.append(
            "Some saved export files are missing locally: "
            + ", ".join(sorted(set(missing_paths)))
            + "."
        )

    restored_state = {
        "run_id": _sanitize_text(payload.get("run_id")),
        "session_id": _sanitize_text(payload.get("session_id")),
        "created_at": _sanitize_text(payload.get("created_at")),
        "updated_at": _sanitize_text(payload.get("updated_at")),
        "user_query": _sanitize_text(payload.get("user_query")),
        "requested_outputs": requested_outputs,
        "workflow_status": _sanitize_text(payload.get("workflow_status")),
        "ui_workflow_status": _sanitize_text(payload.get("ui_workflow_status"))
        or _sanitize_text(payload.get("workflow_status")),
        "routing_decision": _sanitize_text(payload.get("routing_decision")),
        "final_response": _sanitize_text(payload.get("final_response")),
        "content_drafts": content_drafts,
        "partial_outputs": partial_outputs,
        "partial_output_mode": _sanitize_text(payload.get("partial_output_mode")) or "none",
        "image_prompts": [_sanitize_text(item) for item in _safe_list(payload.get("image_prompts", [])) if _sanitize_text(item)],
        "image_outputs": image_outputs,
        "sources": sources,
        "quality_scores": _sanitize_quality_scores(payload.get("quality_scores", {})),
        "export_metadata": export_metadata,
        "warnings": warnings,
        "errors": _sanitize_errors(payload.get("errors", [])),
        "ui_progress_events": _sanitize_progress_events(payload.get("progress_events", [])),
        "status_messages": _sanitize_status_messages(payload.get("status_messages", [])),
        "ui_selected_options": _sanitize_ui_selected_options(payload.get("ui_selected_options", {})),
        "ui_node_statuses": _sanitize_node_statuses(payload.get("ui_node_statuses", {})),
        "prompt_injection_detected": bool(payload.get("prompt_injection_detected", False)),
        "prompt_injection_signals": _sanitize_injection_signals(
            payload.get("prompt_injection_signals", [])
        ),
        "sanitized_user_query": _sanitize_text(payload.get("sanitized_user_query")),
    }
    return restored_state


def to_run_summary(record: Mapping[str, Any]) -> Dict[str, Any]:
    run_id = _sanitize_text(record.get("run_id"))
    session_id = _sanitize_text(record.get("session_id"))
    user_query = _sanitize_text(record.get("user_query"))
    preview = user_query if len(user_query) <= 120 else user_query[:117].rstrip() + "..."
    export_meta = _sanitize_export_metadata(record.get("export_metadata", {}))
    export_available = bool(_safe_dict(export_meta.get("export_paths", {})))
    summary = PersistedRunSummary(
        run_id=run_id,
        session_id=session_id,
        created_at=_sanitize_text(record.get("created_at")),
        updated_at=_sanitize_text(record.get("updated_at")),
        user_query_preview=preview,
        requested_outputs=_sanitize_requested_outputs(record.get("requested_outputs", [])),
        workflow_status=_sanitize_text(record.get("workflow_status")),
        export_available=export_available,
    )
    return asdict(summary)
