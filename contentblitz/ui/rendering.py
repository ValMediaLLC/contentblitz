"""Read-only rendering helpers for safe workflow UI output."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from contentblitz.ui.error_display import normalize_errors_for_display
from contentblitz.ui.progress import normalize_progress_status
from contentblitz.ui.status import (
    apply_optional_node_skips,
    build_initial_node_statuses,
    summarize_workflow_status,
    workflow_requires_clarification,
)

_TERMINAL_FOR_PARTIAL_RENDER = {"completed", "degraded"}


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _source_key(source: Mapping[str, Any], index: int) -> str:
    url = _safe_url(source.get("url"))
    if url:
        return f"url:{url.lower()}"
    title = _safe_text(source.get("title")).lower()
    if title:
        return f"title:{title}"
    return f"idx:{index}"


def _sanitize_source(source: Mapping[str, Any], index: int) -> dict[str, Any]:
    title = _safe_text(source.get("title")) or f"Source {index + 1}"
    url = _safe_url(source.get("url"))
    snippet = _safe_text(source.get("snippet"))
    published_at = _safe_text(source.get("published_at")) or None
    provider = _safe_text(source.get("provider") or source.get("source")) or "unknown"
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
            "id",
            "mime_type",
            "width",
            "height",
            "prompt",
            "revised_prompt",
        ):
            value = raw.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            safe_entry[key] = value

        if "url" in safe_entry and _contains_base64_payload(safe_entry["url"]):
            continue

        raw_error = raw.get("error")
        if raw_error is not None:
            safe_entry["error"] = normalize_errors_for_display(
                [{"message": _safe_text(raw_error), "recoverable": True}]
            )[0]

        if safe_entry:
            safe_outputs.append(safe_entry)

    return safe_outputs


def _node_ready(node_statuses: Mapping[str, str], node_name: str) -> bool:
    return normalize_progress_status(node_statuses.get(node_name, "pending")) in _TERMINAL_FOR_PARTIAL_RENDER


def _quality_warnings(quality_scores: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    for output_type in ("blog", "linkedin"):
        score = _safe_dict(quality_scores.get(output_type, {}))
        validation_status = _safe_text(score.get("validation_status")).lower()
        if validation_status in {"failed", "retry_needed", "unverified"}:
            warnings.append(f"{output_type.title()} quality status: {validation_status}.")
    return warnings


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
    blog_draft = _safe_text(_safe_dict(content_drafts.get("blog", {})).get("body"))
    linkedin_draft = _safe_text(_safe_dict(content_drafts.get("linkedin", {})).get("body"))
    research_report = _safe_text(
        _safe_dict(content_drafts.get("research_report", {})).get("body")
    )
    research_data = _safe_dict(state_snapshot.get("research_data", {}))
    research_summary = _safe_text(
        research_data.get("synthesized_summary") or research_data.get("summary")
    )

    image_prompts: list[str] = []
    if _node_ready(merged_statuses, "image_agent_node"):
        image_prompts = [
            _safe_text(item) for item in _safe_list(state_snapshot.get("image_prompts", [])) if _safe_text(item)
        ]

    image_outputs = []
    if _node_ready(merged_statuses, "image_agent_node"):
        image_outputs = sanitize_image_outputs_for_display(
            _safe_list(state_snapshot.get("image_outputs", []))
        )

    warnings: list[str] = []
    for item in _safe_list(state_snapshot.get("warnings", [])):
        text = _safe_text(item)
        if text:
            warnings.append(text)

    if bool(research_data.get("degraded", False)):
        warnings.append("Research results are degraded and may require manual verification.")

    if any(
        _safe_text(_safe_dict(item).get("status")).lower() == "failed"
        for item in _safe_list(state_snapshot.get("image_outputs", []))
    ):
        warnings.append(
            "Image generation failed in this run, but text outputs may still be usable."
        )

    export_metadata = _safe_dict(state_snapshot.get("export_metadata", {}))
    export_errors = _safe_list(export_metadata.get("error_log", []))
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
    final_response = _safe_text(state_snapshot.get("final_response"))
    if export_errors and final_response:
        warnings.append(
            "Export encountered a non-blocking failure; the final response is still available."
        )

    warnings.extend(_quality_warnings(_safe_dict(state_snapshot.get("quality_scores", {}))))

    partial_outputs = {
        "blog": blog_draft if _node_ready(merged_statuses, "blog_writer_node") else "",
        "linkedin": (
            linkedin_draft if _node_ready(merged_statuses, "linkedin_writer_node") else ""
        ),
        "research": (
            research_report or research_summary
            if (
                _node_ready(merged_statuses, "research_agent_node")
                or _node_ready(merged_statuses, "output_assembler_node")
            )
            else ""
        ),
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
        "sources": dedupe_sources_for_display(state_snapshot.get("sources", [])),
        "errors": normalize_errors_for_display(state_snapshot.get("errors", [])),
        "warnings": list(dict.fromkeys([item for item in warnings if _safe_text(item)])),
        "node_statuses": merged_statuses,
        "export_status": {
            "requested": bool(state_snapshot.get("export_requested", False))
            or bool(_safe_list(export_metadata.get("formats_requested", []))),
            "paths": export_paths,
            "errors": normalize_errors_for_display(export_errors),
            "non_blocking_failure": bool(export_errors) and bool(final_response),
        },
    }
