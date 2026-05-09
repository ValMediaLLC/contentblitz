"""Output assembler node implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _requested_outputs(state: Mapping[str, Any]) -> List[str]:
    outputs = [str(item).strip().lower() for item in _safe_list(state.get("requested_outputs", []))]
    return [item for item in outputs if item]


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _select_text_draft(
    output_type: str,
    content_drafts: Mapping[str, Any],
    best_drafts: Mapping[str, Any],
    quality_scores: Mapping[str, Any],
) -> str:
    draft = _safe_dict(content_drafts.get(output_type, {}))
    current_body = str(draft.get("body", "")).strip()
    current_score = _as_float(_safe_dict(quality_scores.get(output_type, {})).get("composite"), default=-1.0)

    best = _safe_dict(best_drafts.get(output_type, {}))
    best_body = str(best.get("body", "")).strip()
    best_score = _as_float(best.get("composite", best.get("composite_score")), default=-1.0)

    if best_body:
        # Best drafts are preferred when available.
        return best_body
    # Fallback to current draft if best draft is unavailable.
    return current_body


def _source_key(source: Mapping[str, Any], index: int) -> str:
    url = source.get("url")
    if isinstance(url, str) and url.strip():
        return f"url:{url.strip().lower()}"
    title = str(source.get("title", "")).strip().lower()
    if title:
        return f"title:{title}"
    return f"idx:{index}"


def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered_keys: List[str] = []
    best_by_key: Dict[str, Dict[str, Any]] = {}
    score_by_key: Dict[str, float] = {}

    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            continue
        key = _source_key(source, index)
        credibility = _as_float(source.get("credibility_score"), default=0.0)
        if key not in best_by_key:
            ordered_keys.append(key)
            best_by_key[key] = dict(source)
            score_by_key[key] = credibility
            continue
        if credibility > score_by_key.get(key, 0.0):
            best_by_key[key] = dict(source)
            score_by_key[key] = credibility

    return [best_by_key[key] for key in ordered_keys]


def _render_sources_section(deduped_sources: List[Dict[str, Any]]) -> str:
    if not deduped_sources:
        return ""
    lines = ["## Sources"]
    for index, source in enumerate(deduped_sources, start=1):
        title = str(source.get("title", "Source")).strip() or "Source"
        url = source.get("url")
        if isinstance(url, str) and url.strip():
            lines.append(f"[{index}] {title} ({url.strip()})")
        else:
            lines.append(f"[{index}] {title}")
    return "\n".join(lines)


def _render_research_inline_report(state: Mapping[str, Any], deduped_sources: List[Dict[str, Any]]) -> str:
    content_drafts = _safe_dict(state.get("content_drafts", {}))
    report = _safe_dict(content_drafts.get("research_report", {}))
    title = str(report.get("title", "")).strip() or "Research Report"
    body = str(report.get("body", "")).strip()
    sections = [str(item).strip() for item in _safe_list(report.get("sections", [])) if str(item).strip()]

    if not body:
        query = str(state.get("user_query", "")).strip() or "Requested Topic"
        research_data = _safe_dict(state.get("research_data", {}))
        summary = str(research_data.get("synthesized_summary", "")).strip() or str(
            research_data.get("summary", "")
        ).strip()
        if not summary:
            summary = f"Limited research data was available for '{query}'."
        key_facts = [
            str(item).strip() for item in _safe_list(research_data.get("key_facts", [])) if str(item).strip()
        ]
        if not key_facts:
            key_facts = [
                f"Research context generated for '{query}'.",
                "Directional insights were synthesized deterministically.",
                "Additional validation may be required before publication.",
            ]
        body_lines = [summary, "", "Key Facts:"]
        body_lines.extend([f"- {fact}" for fact in key_facts[:5]])
        body = "\n".join(body_lines).strip()

    report_lines = [f"# {title}", "", body]
    if sections:
        report_lines.extend(["", "Sections:"])
        report_lines.extend([f"- {section}" for section in sections])

    sources_block = _render_sources_section(deduped_sources)
    if sources_block:
        report_lines.extend(["", sources_block])
    return "\n".join(report_lines).strip()


def _quality_warnings(quality_scores: Mapping[str, Any]) -> Tuple[List[str], bool]:
    warnings: List[str] = []
    partial = False
    for output_type in ("blog", "linkedin"):
        score = _safe_dict(quality_scores.get(output_type, {}))
        status = str(score.get("validation_status", "")).strip().lower()
        if not status:
            continue
        if status in {"failed", "unverified", "retry_needed"}:
            warnings.append(f"{output_type.title()} quality status: {status}.")
        if status in {"failed", "unverified"}:
            partial = True
    return warnings, partial


def _image_summary(image_outputs: List[Dict[str, Any]], errors: List[Dict[str, Any]]) -> Tuple[str, bool]:
    success_assets: List[str] = []
    failed = False

    for output in image_outputs:
        if not isinstance(output, Mapping):
            continue
        status = str(output.get("status", "")).strip().lower()
        if status == "success":
            url = output.get("url")
            asset = str(url).strip() if isinstance(url, str) and url.strip() else str(output.get("id", "")).strip()
            if asset:
                success_assets.append(asset)
        elif status == "failed":
            failed = True

    for error in errors:
        if not isinstance(error, Mapping):
            continue
        if (
            str(error.get("agent", "")).strip() == "image_agent"
            and bool(error.get("recoverable", False))
        ):
            failed = True

    if success_assets:
        lines = ["## Image Assets"]
        lines.extend([f"- {asset}" for asset in success_assets])
        return "\n".join(lines), failed
    return "", failed


def _assemble_image_output(
    image_prompts: List[str],
    image_outputs: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
) -> str:
    success_assets: List[str] = []
    failed = False

    for output in image_outputs:
        if not isinstance(output, Mapping):
            continue
        status = str(output.get("status", "")).strip().lower()
        if status == "success":
            url = output.get("url")
            if isinstance(url, str) and url.strip():
                success_assets.append(url.strip())
            else:
                image_id = str(output.get("id", "")).strip()
                if image_id:
                    success_assets.append(image_id)
        elif status == "failed":
            failed = True

    for error in errors:
        if not isinstance(error, Mapping):
            continue
        if (
            str(error.get("agent", "")).strip() == "image_agent"
            and bool(error.get("recoverable", False))
        ):
            failed = True

    if success_assets:
        return "Image assets: " + ", ".join(success_assets)
    if failed:
        prompt_hint = image_prompts[-1] if image_prompts else "the requested concept"
        return (
            "Image generation is temporarily unavailable for "
            f"'{prompt_hint}'. Recoverable failure recorded."
        )
    if image_prompts:
        return f"Image concept prepared: {image_prompts[-1]}"
    return ""


def _assemble_research_output(state: Mapping[str, Any]) -> str:
    research_data = _safe_dict(state.get("research_data", {}))
    summary = str(research_data.get("synthesized_summary", "")).strip() or str(
        research_data.get("summary", "")
    ).strip()
    if not summary:
        query = str(state.get("user_query", "")).strip() or "requested topic"
        summary = f"Research summary is limited for '{query}'."

    key_facts = [
        str(item).strip()
        for item in _safe_list(research_data.get("key_facts", []))
        if str(item).strip()
    ]
    if key_facts:
        facts_text = "; ".join(key_facts[:3])
        return f"{summary} Key facts: {facts_text}."
    return summary


def output_assembler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble final response deterministically from existing state only."""
    outputs = _requested_outputs(state)
    content_drafts = _safe_dict(state.get("content_drafts", {}))
    best_drafts = _safe_dict(state.get("best_drafts", {}))
    quality_scores = _safe_dict(state.get("quality_scores", {}))
    errors = deepcopy(_safe_list(state.get("errors", [])))
    sources = deepcopy([item for item in _safe_list(state.get("sources", [])) if isinstance(item, Mapping)])
    image_prompts = deepcopy(_safe_list(state.get("image_prompts", [])))
    image_outputs = [
        dict(item) for item in _safe_list(state.get("image_outputs", [])) if isinstance(item, Mapping)
    ]

    deduped_sources = _dedupe_sources([dict(item) for item in sources])

    sections: List[str] = []
    assembled_outputs: Dict[str, Any] = {}
    usable_content = False
    partial_success = False

    if outputs == ["research"]:
        research_report = _render_research_inline_report(state, deduped_sources)
        if research_report:
            sections.append(research_report)
            assembled_outputs["research"] = _assemble_research_output(state)
            usable_content = True
    else:
        if "blog" in outputs:
            blog_body = _select_text_draft("blog", content_drafts, best_drafts, quality_scores)
            if blog_body:
                sections.append("## Blog Draft\n" + blog_body)
                assembled_outputs["blog"] = blog_body
                usable_content = True

        if "linkedin" in outputs:
            linkedin_body = _select_text_draft("linkedin", content_drafts, best_drafts, quality_scores)
            if linkedin_body:
                sections.append("## LinkedIn Draft\n" + linkedin_body)
                assembled_outputs["linkedin"] = linkedin_body
                usable_content = True

        if "research" in outputs:
            report = _safe_dict(content_drafts.get("research_report", {}))
            report_title = str(report.get("title", "")).strip() or "Research Report"
            report_body = str(report.get("body", "")).strip()
            if report_body:
                sections.append(f"## {report_title}\n{report_body}")
                assembled_outputs["research"] = report_body
                usable_content = True

    image_section, image_failed = _image_summary(
        image_outputs=image_outputs,
        errors=[dict(item) for item in errors if isinstance(item, Mapping)],
    )
    image_output_text = _assemble_image_output(
        image_prompts=image_prompts,
        image_outputs=image_outputs,
        errors=[dict(item) for item in errors if isinstance(item, Mapping)],
    )
    if image_output_text:
        assembled_outputs["image"] = image_output_text
    if image_section:
        sections.append(image_section)
        usable_content = True
    if image_failed:
        sections.append("Warning: Image generation encountered a recoverable failure.")
        partial_success = True
        if "image" in outputs:
            usable_content = True

    quality_warnings, quality_partial = _quality_warnings(quality_scores)
    if quality_warnings:
        warning_lines = ["## Quality Warnings"]
        warning_lines.extend([f"- {warning}" for warning in quality_warnings])
        sections.append("\n".join(warning_lines))
    if quality_partial:
        partial_success = True

    cost_controls = _safe_dict(state.get("cost_controls", {}))
    if bool(cost_controls.get("budget_exceeded", False)):
        sections.append("Notice: Session budget was exceeded during generation.")
        partial_success = True

    sources_block = _render_sources_section(deduped_sources)
    if sources_block and outputs != ["research"]:
        sections.append(sources_block)

    if not usable_content:
        final_response = (
            "Unable to assemble usable content for the requested outputs. "
            "Please refine your prompt and try again."
        )
        workflow_status = "failed"
    else:
        final_response = "\n\n".join([section.strip() for section in sections if section.strip()]).strip()
        if not final_response:
            final_response = "Content assembled, but response formatting was empty."
            workflow_status = "failed"
        else:
            workflow_status = "partial_success" if partial_success else "success"

    export_metadata = _safe_dict(state.get("export_metadata", {}))
    formats_requested = _safe_list(export_metadata.get("formats_requested", []))
    export_requested = bool(state.get("export_requested", False)) or bool(formats_requested)

    return {
        "final_response": final_response,
        "assembled_outputs": assembled_outputs,
        "workflow_status": workflow_status,
        "export_requested": export_requested,
    }
