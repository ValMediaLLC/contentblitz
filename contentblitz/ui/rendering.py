"""Read-only rendering helpers for safe workflow UI output."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from contentblitz.core.warnings import (
    IMAGE_RECOVERABLE_WARNING,
    TEXT_FALLBACK_WARNING,
    TOP_LEVEL_PROVIDER_WARNING,
    dedupe_user_warnings,
)
from contentblitz.safety.output_sanitizer import (
    sanitize_markdown_output,
    sanitize_plain_output,
)
from contentblitz.ui.error_display import normalize_errors_for_display
from contentblitz.ui.progress import normalize_progress_status
from contentblitz.ui.status import (
    apply_optional_node_skips,
    build_initial_node_statuses,
    summarize_workflow_status,
    workflow_requires_clarification,
)

_TERMINAL_FOR_PARTIAL_RENDER = {"completed", "degraded"}
_PERFORMANCE_TERMINAL_STATUSES = {"completed", "degraded", "failed", "skipped"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url:
        return None
    if url.startswith(("http://", "https://")):
        return url
    return None


def _safe_local_image_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw_path = value.strip()
    if not raw_path:
        return None
    lowered = raw_path.lower()
    if lowered.startswith("data:image/") or "base64" in lowered:
        return None
    try:
        path = Path(raw_path)
        resolved = (
            path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
        )
        if not resolved.exists() or not resolved.is_file():
            return None
        if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return None
        try:
            return resolved.relative_to(Path.cwd().resolve()).as_posix()
        except Exception:
            return resolved.as_posix()
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _normalize_format_list(value: Any) -> list[str]:
    normalized: list[str] = []
    for item in _safe_list(value):
        token = _safe_text(item).lower()
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _failed_export_formats(export_metadata: Mapping[str, Any]) -> list[str]:
    explicit_failed = _normalize_format_list(
        export_metadata.get("failed_export_formats")
    )
    if explicit_failed:
        return explicit_failed
    export_status = _safe_dict(export_metadata.get("export_status", {}))
    return [
        _safe_text(fmt).lower()
        for fmt, status in export_status.items()
        if _safe_text(fmt) and _safe_text(status).lower() == "failed"
    ]


def _is_warning_export_log_entry(entry: Mapping[str, Any]) -> bool:
    code = _safe_text(entry.get("code")).lower()
    if code.endswith("_warning") or code == "warning":
        return True
    message = _safe_text(entry.get("message")).lower()
    return "warning" in message and "failed" not in message


def _export_failure_count(export_metadata: Mapping[str, Any]) -> int:
    explicit = _safe_int(export_metadata.get("export_error_count"), default=-1)
    if explicit >= 0:
        return explicit
    failed_formats = _failed_export_formats(export_metadata)
    if failed_formats:
        return len(failed_formats)
    return sum(
        1
        for item in _safe_list(export_metadata.get("error_log", []))
        if isinstance(item, Mapping) and not _is_warning_export_log_entry(item)
    )


def _export_warning_count(export_metadata: Mapping[str, Any]) -> int:
    explicit = _safe_int(export_metadata.get("export_warning_count"), default=-1)
    if explicit >= 0:
        return explicit
    return sum(
        1
        for item in _safe_list(export_metadata.get("error_log", []))
        if isinstance(item, Mapping) and _is_warning_export_log_entry(item)
    )


def _sanitized_plain(value: Any) -> str:
    sanitized, _ = sanitize_plain_output(value)
    return sanitized


def _source_key(source: Mapping[str, Any], index: int) -> str:
    url = _safe_url(source.get("url"))
    if url:
        return f"url:{url.lower()}"
    title = _safe_text(source.get("title")).lower()
    if title:
        return f"title:{title}"
    return f"idx:{index}"


def _sanitize_source(source: Mapping[str, Any], index: int) -> dict[str, Any]:
    title, _ = sanitize_plain_output(_safe_text(source.get("title")))
    title = title or f"Source {index + 1}"
    url = _safe_url(source.get("url"))
    snippet, _ = sanitize_plain_output(_safe_text(source.get("snippet")))
    published_at, _ = sanitize_plain_output(_safe_text(source.get("published_at")))
    published_at = published_at or None
    provider, _ = sanitize_plain_output(
        _safe_text(source.get("provider") or source.get("source"))
    )
    provider = provider or "unknown"
    citation_available = bool(url) and bool(source.get("citation_available", False))
    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "source": provider,
        "published_at": published_at,
        "citation_available": citation_available,
        "credibility_score": _safe_float(source.get("credibility_score"), default=0.0),
    }


def dedupe_sources_for_display(sources: Any) -> list[dict[str, Any]]:
    """Deduplicate sources for display using URL/title priority."""
    if not isinstance(sources, list):
        return []

    best_by_key: dict[str, dict[str, Any]] = {}
    score_by_key: dict[str, float] = {}
    order: list[str] = []

    for index, raw in enumerate(sources):
        if not isinstance(raw, Mapping):
            continue
        sanitized = _sanitize_source(raw, index)
        key = _source_key(sanitized, index)
        score = _safe_float(sanitized.get("credibility_score"), default=0.0)

        if key not in best_by_key:
            best_by_key[key] = sanitized
            score_by_key[key] = score
            order.append(key)
            continue

        if score > score_by_key.get(key, 0.0):
            best_by_key[key] = sanitized
            score_by_key[key] = score

    return [best_by_key[key] for key in order]


def _contains_base64_payload(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip().lower()
    if not candidate:
        return False
    if candidate.startswith("data:image/"):
        return True
    if "base64" in candidate:
        return True
    return False


def sanitize_image_outputs_for_display(image_outputs: Any) -> list[dict[str, Any]]:
    """Return image outputs with any base64 payloads removed/rejected."""
    if not isinstance(image_outputs, list):
        return []

    safe_outputs: list[dict[str, Any]] = []
    for raw in image_outputs:
        if not isinstance(raw, Mapping):
            continue

        raw_url = raw.get("url")
        if _contains_base64_payload(raw_url):
            continue

        safe_entry: dict[str, Any] = {}
        for key in (
            "status",
            "provider",
            "url",
            "local_path",
            "id",
            "renderable",
            "mime_type",
            "width",
            "height",
            "prompt",
            "revised_prompt",
        ):
            value = raw.get(key)
            if value is None:
                continue
            if key in {"width", "height"}:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    safe_entry[key] = value
                continue
            if key == "renderable":
                if isinstance(value, bool):
                    safe_entry[key] = value
                continue

            sanitized_value = _sanitized_plain(value)
            if not sanitized_value:
                continue

            if key == "url":
                if _contains_base64_payload(sanitized_value):
                    continue
                safe_url = _safe_url(sanitized_value)
                if not safe_url:
                    continue
                safe_entry[key] = safe_url
                continue

            if key == "local_path":
                safe_local_path = _safe_local_image_path(sanitized_value)
                if not safe_local_path:
                    continue
                safe_entry[key] = safe_local_path
                continue

            safe_entry[key] = sanitized_value

        if "url" in safe_entry and _contains_base64_payload(safe_entry["url"]):
            continue
        if "local_path" in safe_entry and _contains_base64_payload(
            safe_entry["local_path"]
        ):
            continue

        if "renderable" not in safe_entry:
            safe_entry["renderable"] = bool(
                safe_entry.get("url") or safe_entry.get("local_path")
            )

        raw_error = raw.get("error")
        if raw_error is not None:
            recoverable = True
            code = ""
            provider = ""
            message_value: Any = raw_error
            if isinstance(raw_error, Mapping):
                recoverable = bool(raw_error.get("recoverable", True))
                code = _sanitized_plain(raw_error.get("code"))
                provider = _sanitized_plain(raw_error.get("provider"))
                message_value = raw_error.get("message")

            safe_message = _sanitized_plain(message_value)
            if not safe_message:
                safe_message = (
                    "Image generation encountered a recoverable issue."
                    if recoverable
                    else "Image generation failed safely."
                )
            normalized_error_payload: dict[str, Any] = {
                "message": safe_message,
                "recoverable": recoverable,
            }
            if code:
                normalized_error_payload["code"] = code
            if provider:
                normalized_error_payload["provider"] = provider
            safe_entry["error"] = normalize_errors_for_display(
                [normalized_error_payload]
            )[0]

        if safe_entry:
            safe_outputs.append(safe_entry)

    return safe_outputs


def _node_ready(node_statuses: Mapping[str, str], node_name: str) -> bool:
    return (
        normalize_progress_status(node_statuses.get(node_name, "pending"))
        in _TERMINAL_FOR_PARTIAL_RENDER
    )


def _quality_warnings(quality_scores: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    for output_type in ("blog", "linkedin"):
        score = _safe_dict(quality_scores.get(output_type, {}))
        validation_status = _safe_text(score.get("validation_status")).lower()
        if validation_status in {"failed", "retry_needed", "unverified", "degraded"}:
            warnings.append(
                f"{output_type.title()} quality status: {validation_status}."
            )
    return warnings


def _is_fallback_draft(draft: Mapping[str, Any]) -> bool:
    if bool(draft.get("fallback_generated", False)):
        return True
    if bool(draft.get("degraded_generation", False)):
        return True
    generation_status = _safe_text(draft.get("generation_status")).lower()
    if generation_status in {"fallback_degraded", "fallback_generated"}:
        return True
    provider_status = _safe_text(draft.get("provider_status")).lower()
    return provider_status == "degraded"


def _has_text_generation_degraded(content_drafts: Mapping[str, Any]) -> bool:
    for channel in ("blog", "linkedin"):
        if _is_fallback_draft(_safe_dict(content_drafts.get(channel, {}))):
            return True
    return False


def _derive_usage_summary(
    *,
    state_snapshot: Mapping[str, Any],
    node_statuses: Mapping[str, str],
    ui_workflow_status: str,
    sources_returned: int,
) -> dict[str, Any]:
    usage_metrics = _safe_dict(state_snapshot.get("usage_metrics", {}))
    cost_controls = _safe_dict(state_snapshot.get("cost_controls", {}))
    retry_counts = _safe_dict(state_snapshot.get("retry_counts", {}))
    content_drafts = _safe_dict(state_snapshot.get("content_drafts", {}))
    export_metadata = _safe_dict(state_snapshot.get("export_metadata", {}))
    raw_image_outputs = _safe_list(state_snapshot.get("image_outputs", []))
    research_data = _safe_dict(state_snapshot.get("research_data", {}))
    raw_errors = _safe_list(state_snapshot.get("errors", []))

    tokens_used = max(
        0,
        _safe_int(
            usage_metrics.get("estimated_total_tokens"),
            _safe_int(cost_controls.get("tokens_used_this_session"), 0),
        ),
    )
    estimated_tokens_out = max(
        0,
        _safe_int(usage_metrics.get("estimated_tokens_out"), tokens_used),
    )
    estimated_tokens_in = max(
        0,
        _safe_int(
            usage_metrics.get("estimated_tokens_in"),
            max(0, tokens_used - estimated_tokens_out),
        ),
    )
    estimated_tokens_total = max(0, estimated_tokens_in + estimated_tokens_out)

    text_generation_calls = max(
        0, _safe_int(usage_metrics.get("text_generation_calls"), 0)
    )
    if text_generation_calls <= 0:
        blog_version = max(
            0,
            _safe_int(_safe_dict(content_drafts.get("blog", {})).get("version"), 0),
        )
        linkedin_version = max(
            0,
            _safe_int(_safe_dict(content_drafts.get("linkedin", {})).get("version"), 0),
        )
        text_generation_calls = blog_version + linkedin_version
        if text_generation_calls <= 0 and estimated_tokens_total > 0:
            text_generation_calls = 1

    search_queries = max(
        0,
        _safe_int(
            usage_metrics.get("search_queries"),
            _safe_int(cost_controls.get("search_queries_used_this_session"), 0),
        ),
    )

    image_generation_failures = sum(
        1
        for item in raw_image_outputs
        if _safe_text(_safe_dict(item).get("status")).lower() == "failed"
    )
    image_generation_requests = max(
        0,
        _safe_int(
            usage_metrics.get("image_generation_requests"),
            max(
                _safe_int(cost_controls.get("image_generations_used_this_session"), 0),
                len(raw_image_outputs),
            ),
        ),
    )

    retry_attempts = max(
        0,
        _safe_int(
            usage_metrics.get("retry_attempts"),
            _safe_int(
                cost_controls.get("total_retries_used_this_session"),
                sum(max(0, _safe_int(value, 0)) for value in retry_counts.values()),
            ),
        ),
    )

    degraded_node_count = sum(
        1
        for status in node_statuses.values()
        if _safe_text(status).lower() == "degraded"
    )
    recoverable_error_count = sum(
        1 for item in raw_errors if bool(_safe_dict(item).get("recoverable", False))
    )
    degraded_operations = max(
        0,
        _safe_int(
            usage_metrics.get("degraded_operations"),
            degraded_node_count
            + image_generation_failures
            + recoverable_error_count
            + (1 if bool(research_data.get("degraded", False)) else 0),
        ),
    )

    export_status = _safe_dict(export_metadata.get("export_status", {}))
    export_paths = _safe_dict(export_metadata.get("export_paths", {}))
    export_generation_count = max(
        0,
        _safe_int(
            usage_metrics.get("export_generation_count"),
            max(
                len(export_paths),
                sum(
                    1
                    for status in export_status.values()
                    if _safe_text(status).lower() == "completed"
                ),
            ),
        ),
    )

    estimated_total_operations = max(
        0,
        _safe_int(
            usage_metrics.get("estimated_total_operations"),
            text_generation_calls
            + search_queries
            + image_generation_requests
            + retry_attempts
            + export_generation_count,
        ),
    )

    estimated_workflow_cost_level = _safe_text(
        usage_metrics.get("estimated_workflow_cost_level")
    ).lower()
    if estimated_workflow_cost_level not in {"low", "medium", "high"}:
        if estimated_tokens_total >= 7000 or estimated_total_operations >= 20:
            estimated_workflow_cost_level = "high"
        elif estimated_tokens_total >= 2000 or estimated_total_operations >= 8:
            estimated_workflow_cost_level = "medium"
        else:
            estimated_workflow_cost_level = "low"

    budget_state = _safe_text(usage_metrics.get("budget_state")).lower()
    if budget_state not in {"normal", "degraded", "limited", "budget_exceeded"}:
        if bool(cost_controls.get("budget_exceeded", False)):
            budget_state = "budget_exceeded"
        else:
            limited = False
            token_budget = max(
                0, _safe_int(cost_controls.get("token_budget_per_session"), 0)
            )
            if token_budget > 0 and estimated_tokens_total >= int(token_budget * 0.9):
                limited = True
            search_cap = max(
                0, _safe_int(cost_controls.get("search_query_cap_per_session"), 0)
            )
            if search_cap > 0 and search_queries >= search_cap:
                limited = True
            image_cap = max(
                0,
                _safe_int(cost_controls.get("image_generation_cap_per_session"), 0),
            )
            if image_cap > 0 and image_generation_requests >= image_cap:
                limited = True
            retry_cap = max(
                0, _safe_int(cost_controls.get("max_total_retries_per_session"), 0)
            )
            if retry_cap > 0 and retry_attempts >= retry_cap:
                limited = True

            if limited:
                budget_state = "limited"
            elif degraded_operations > 0 or ui_workflow_status == "partial_success":
                budget_state = "degraded"
            else:
                budget_state = "normal"

    return {
        "text_generation_calls": text_generation_calls,
        "estimated_tokens_in": estimated_tokens_in,
        "estimated_tokens_out": estimated_tokens_out,
        "search_queries": search_queries,
        "sources_returned": max(0, _safe_int(sources_returned, 0)),
        "image_generation_requests": image_generation_requests,
        "image_generation_failures": image_generation_failures,
        "retry_attempts": retry_attempts,
        "degraded_operations": degraded_operations,
        "export_generation_count": export_generation_count,
        "estimated_total_operations": estimated_total_operations,
        "estimated_workflow_cost_level": estimated_workflow_cost_level,
        "budget_state": budget_state,
    }


def _derive_performance_summary(state_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    raw_events = state_snapshot.get("ui_progress_events")
    if not isinstance(raw_events, list):
        raw_events = state_snapshot.get("progress_events", [])
    if not isinstance(raw_events, list):
        return {}

    per_node: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []

    for raw_event in raw_events:
        if not isinstance(raw_event, Mapping):
            continue
        node_name = _safe_text(raw_event.get("node_name"))
        if not node_name:
            continue
        status = normalize_progress_status(
            _safe_text(raw_event.get("status", "pending"))
        )
        if node_name not in per_node:
            node_order.append(node_name)
            per_node[node_name] = {"node_name": node_name, "status": status}
        else:
            per_node[node_name]["status"] = status

        safe_metadata = _safe_dict(raw_event.get("safe_metadata", {}))
        duration_ms = _safe_int(safe_metadata.get("duration_ms"), default=-1)
        provider_latency_ms = _safe_int(
            safe_metadata.get("provider_latency_ms"),
            default=-1,
        )

        if duration_ms >= 0:
            per_node[node_name]["duration_ms"] = duration_ms
        if provider_latency_ms >= 0:
            per_node[node_name]["provider_latency_ms"] = provider_latency_ms
        if "cache_hit" in safe_metadata:
            per_node[node_name]["cache_hit"] = bool(safe_metadata.get("cache_hit"))
        provider = _safe_text(safe_metadata.get("provider")).lower()
        model = _safe_text(safe_metadata.get("model"))
        if provider:
            per_node[node_name]["provider"] = provider
        if model:
            per_node[node_name]["model"] = model

    if not per_node:
        return {}

    ordered_nodes = [per_node[name] for name in node_order]
    timed_nodes = [
        node
        for node in ordered_nodes
        if isinstance(node.get("duration_ms"), int) and node.get("duration_ms", -1) >= 0
    ]
    provider_latency_nodes = [
        node
        for node in ordered_nodes
        if isinstance(node.get("provider_latency_ms"), int)
        and node.get("provider_latency_ms", -1) >= 0
    ]
    total_duration_ms = sum(int(node.get("duration_ms", 0)) for node in timed_nodes)
    total_provider_latency_ms = sum(
        int(node.get("provider_latency_ms", 0)) for node in provider_latency_nodes
    )

    terminal_node_count = sum(
        1
        for node in ordered_nodes
        if _safe_text(node.get("status")).lower() in _PERFORMANCE_TERMINAL_STATUSES
    )

    return {
        "executed_node_count": len(ordered_nodes),
        "terminal_node_count": terminal_node_count,
        "timed_node_count": len(timed_nodes),
        "total_duration_ms": total_duration_ms,
        "average_duration_ms": (
            int(total_duration_ms / len(timed_nodes)) if timed_nodes else 0
        ),
        "provider_latency_total_ms": total_provider_latency_ms,
        "timed_provider_node_count": len(provider_latency_nodes),
        "nodes": ordered_nodes,
    }


def build_render_payload(
    *,
    state: Mapping[str, Any],
    node_statuses: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build a read-only render contract for Streamlit components.

    This helper intentionally does not mutate orchestration state.
    """
    state_snapshot = deepcopy(dict(state))
    merged_statuses = build_initial_node_statuses()
    if isinstance(node_statuses, Mapping):
        for key, value in dict(node_statuses).items():
            if key in merged_statuses:
                merged_statuses[key] = normalize_progress_status(str(value))
    merged_statuses = apply_optional_node_skips(
        state=state_snapshot,
        node_statuses=merged_statuses,
    )
    clarification_required = workflow_requires_clarification(
        state=state_snapshot,
        node_statuses=merged_statuses,
    )
    ui_workflow_status = summarize_workflow_status(
        merged_statuses,
        workflow_status=_safe_text(state_snapshot.get("workflow_status")),
        clarification_required=clarification_required,
    )

    content_drafts = _safe_dict(state_snapshot.get("content_drafts", {}))
    text_generation_degraded = _has_text_generation_degraded(content_drafts)
    unsafe_content_removed = False
    blog_draft, blog_changed = sanitize_markdown_output(
        _safe_text(_safe_dict(content_drafts.get("blog", {})).get("body"))
    )
    linkedin_draft, linkedin_changed = sanitize_markdown_output(
        _safe_text(_safe_dict(content_drafts.get("linkedin", {})).get("body"))
    )
    research_report, research_changed = sanitize_markdown_output(
        _safe_text(_safe_dict(content_drafts.get("research_report", {})).get("body"))
    )
    unsafe_content_removed = (
        unsafe_content_removed or blog_changed or linkedin_changed or research_changed
    )
    research_data = _safe_dict(state_snapshot.get("research_data", {}))
    research_summary, summary_changed = sanitize_markdown_output(
        _safe_text(
            research_data.get("synthesized_summary") or research_data.get("summary")
        )
    )
    unsafe_content_removed = unsafe_content_removed or summary_changed

    image_prompts: list[str] = []
    if _node_ready(merged_statuses, "image_agent_node"):
        for item in _safe_list(state_snapshot.get("image_prompts", [])):
            clean = _sanitized_plain(_safe_text(item))
            if clean:
                image_prompts.append(clean)

    image_outputs = []
    if _node_ready(merged_statuses, "image_agent_node"):
        image_outputs = sanitize_image_outputs_for_display(
            _safe_list(state_snapshot.get("image_outputs", []))
        )

    warnings: list[str] = []
    for item in _safe_list(state_snapshot.get("warnings", [])):
        text, changed = sanitize_plain_output(item)
        unsafe_content_removed = unsafe_content_removed or changed
        if text:
            warnings.append(text)

    if bool(research_data.get("degraded", False)):
        warnings.append(
            "Research results are degraded and may require manual verification."
        )

    if any(
        _safe_text(_safe_dict(item).get("status")).lower() == "failed"
        for item in _safe_list(state_snapshot.get("image_outputs", []))
    ):
        warnings.append(IMAGE_RECOVERABLE_WARNING)
    image_generation_degraded = any(
        _safe_text(_safe_dict(item).get("status")).lower() in {"failed", "degraded"}
        for item in _safe_list(state_snapshot.get("image_outputs", []))
    )
    if text_generation_degraded:
        warnings.append(TEXT_FALLBACK_WARNING)
    if text_generation_degraded or image_generation_degraded:
        warnings.append(TOP_LEVEL_PROVIDER_WARNING)

    export_metadata = _safe_dict(state_snapshot.get("export_metadata", {}))
    export_errors = _safe_list(export_metadata.get("error_log", []))
    export_warning_count = _export_warning_count(export_metadata)
    export_error_count = _export_failure_count(export_metadata)
    failed_export_formats = _failed_export_formats(export_metadata)
    requested_export_formats = _normalize_format_list(
        export_metadata.get("requested_export_formats")
        or export_metadata.get("formats_requested", [])
    )
    completed_export_formats = _normalize_format_list(
        export_metadata.get("completed_export_formats", [])
    )
    if not completed_export_formats:
        completed_export_formats = [
            _safe_text(fmt).lower()
            for fmt, status in _safe_dict(
                export_metadata.get("export_status", {})
            ).items()
            if _safe_text(fmt) and _safe_text(status).lower() == "completed"
        ]
    raw_export_paths = _safe_dict(export_metadata.get("export_paths", {}))
    export_paths: dict[str, str] = {}
    missing_export_formats: list[str] = []
    for fmt, raw_path in raw_export_paths.items():
        fmt_name = _safe_text(fmt).lower()
        path_text = _safe_text(raw_path)
        if not fmt_name or not path_text:
            continue
        if Path(path_text).exists():
            export_paths[fmt_name] = path_text
        else:
            missing_export_formats.append(fmt_name)
    if missing_export_formats:
        warnings.append(
            "Some saved export files are missing locally: "
            + ", ".join(sorted(set(missing_export_formats)))
            + "."
        )
    display_sources = dedupe_sources_for_display(state_snapshot.get("sources", []))
    usage_summary = _derive_usage_summary(
        state_snapshot=state_snapshot,
        node_statuses=merged_statuses,
        ui_workflow_status=ui_workflow_status,
        sources_returned=len(display_sources),
    )
    performance_summary = _derive_performance_summary(state_snapshot)
    budget_state = _safe_text(usage_summary.get("budget_state")).lower()
    if budget_state == "degraded":
        warnings.append(
            "Research results used fallback mode due to limited provider availability."
        )
    elif budget_state == "limited":
        warnings.append(
            "Workflow is operating in limited mode. Some outputs may be reduced."
        )
    elif budget_state == "budget_exceeded":
        warnings.append(
            "Workflow usage limits were reached. Some outputs may be reduced."
        )

    final_response = _safe_text(state_snapshot.get("final_response"))
    final_response, final_changed = sanitize_markdown_output(final_response)
    unsafe_content_removed = unsafe_content_removed or final_changed
    if export_error_count > 0 and final_response:
        warnings.append(
            "Export encountered a non-blocking failure; "
            "the final response is still available."
        )
    elif export_warning_count > 0:
        warnings.append("Export completed with non-blocking warnings.")

    warnings.extend(
        _quality_warnings(_safe_dict(state_snapshot.get("quality_scores", {})))
    )
    if unsafe_content_removed:
        warnings.append("Unsafe content was removed before rendering.")

    persisted_ui_statuses = _safe_dict(state_snapshot.get("ui_node_statuses", {}))

    def _historical_node_ready(node_name: str) -> bool:
        raw_status = _safe_text(persisted_ui_statuses.get(node_name))
        if not raw_status:
            return False
        status = normalize_progress_status(raw_status, invalid_fallback="failed")
        return status in _TERMINAL_FOR_PARTIAL_RENDER

    blog_ready_for_display = _node_ready(
        merged_statuses, "blog_writer_node"
    ) or _historical_node_ready("blog_writer_node")
    linkedin_ready_for_display = _node_ready(
        merged_statuses, "linkedin_writer_node"
    ) or _historical_node_ready("linkedin_writer_node")
    research_ready_for_display = (
        _node_ready(merged_statuses, "research_agent_node")
        or _node_ready(merged_statuses, "output_assembler_node")
        or _historical_node_ready("research_agent_node")
        or _historical_node_ready("output_assembler_node")
    )

    partial_outputs = {
        "blog": blog_draft if blog_ready_for_display else "",
        "linkedin": linkedin_draft if linkedin_ready_for_display else "",
        "research": (research_report or research_summary)
        if research_ready_for_display
        else "",
    }
    partial_sections: list[dict[str, str]] = []
    partial_labels = {
        "blog": "Blog Draft",
        "linkedin": "LinkedIn Draft",
        "research": "Research Summary / Research Report",
    }
    for key in ("blog", "linkedin", "research"):
        content = _safe_text(partial_outputs.get(key))
        if not content:
            continue
        partial_sections.append(
            {"key": key, "label": partial_labels[key], "content": content}
        )

    partial_mode = "none"
    if len(partial_sections) > 1:
        partial_mode = "multi_output"
    elif len(partial_sections) == 1:
        partial_mode = f"{partial_sections[0]['key']}_only"

    return {
        "workflow_status": ui_workflow_status,
        "raw_workflow_status": _safe_text(state_snapshot.get("workflow_status")),
        "final_response": final_response,
        "partial_outputs": partial_outputs,
        "partial_output_mode": partial_mode,
        "partial_output_sections": partial_sections,
        "image_prompts": image_prompts,
        "image_outputs": image_outputs,
        "sources": display_sources,
        "errors": normalize_errors_for_display(state_snapshot.get("errors", [])),
        "warnings": dedupe_user_warnings(
            [item for item in warnings if _safe_text(item)]
        ),
        "node_statuses": merged_statuses,
        "usage_summary": usage_summary,
        "performance_summary": performance_summary,
        "export_status": {
            "requested": bool(state_snapshot.get("export_requested", False))
            or bool(_safe_list(export_metadata.get("formats_requested", []))),
            "paths": export_paths,
            "errors": normalize_errors_for_display(export_errors),
            "requested_formats": requested_export_formats,
            "completed_formats": completed_export_formats,
            "failed_formats": failed_export_formats,
            "export_warning_count": export_warning_count,
            "export_error_count": export_error_count,
            "non_blocking_failure": export_error_count > 0 and bool(final_response),
        },
        "provider_status": {
            "text_generation": "degraded" if text_generation_degraded else "completed",
            "image_generation": (
                "degraded" if image_generation_degraded else "completed"
            ),
            "search": (
                "degraded"
                if bool(
                    _safe_dict(state_snapshot.get("research_data", {})).get(
                        "degraded", False
                    )
                )
                else "completed"
            ),
            "export": "degraded" if export_error_count > 0 else "completed",
        },
        "degradation_metadata": {
            "text_generation_degraded": text_generation_degraded,
            "image_generation_degraded": image_generation_degraded,
            "fallback_content_used": text_generation_degraded,
            "real_generation_succeeded": not text_generation_degraded,
            "provider_failure_reason": (
                _safe_text(
                    _safe_dict(content_drafts.get("blog", {})).get(
                        "provider_failure_reason"
                    )
                )
                or _safe_text(
                    _safe_dict(content_drafts.get("linkedin", {})).get(
                        "provider_failure_reason"
                    )
                )
            ),
        },
    }
