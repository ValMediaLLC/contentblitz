"""Export node implementation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping

from contentblitz.tools.exports.docx import (
    build_docx_document_bytes_from_text,
    build_docx_export_document,
)
from contentblitz.tools.exports.filenames import (
    resolve_docx_export_path,
    resolve_export_dir,
    resolve_html_export_path,
    resolve_markdown_export_path,
    resolve_pdf_export_path,
)
from contentblitz.tools.exports.html import (
    build_html_export_document,
    sanitize_html_content,
)
from contentblitz.tools.exports.markdown import (
    build_markdown_export_document,
    sanitize_markdown_content,
)
from contentblitz.tools.exports.pdf import (
    build_pdf_document_bytes_from_text,
    build_pdf_export_document,
)
from contentblitz.tools.exports.validation import (
    normalize_validation_result,
    validate_docx_export,
    validate_html_export,
    validate_markdown_export,
    validate_pdf_export,
)

_SUPPORTED_EXPORT_FORMATS = {"markdown", "html", "pdf", "docx"}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_requested_formats(raw_formats: List[Any]) -> List[str]:
    normalized: List[str] = []
    for item in raw_formats:
        token = _safe_text(item).lower()
        if not token:
            continue
        if token in {"md", "markdown"}:
            token = "markdown"
        elif token == "word":
            token = "docx"
        if token not in normalized:
            normalized.append(token)
    return normalized


def _is_warning_diagnostic(entry: Mapping[str, Any]) -> bool:
    code = _safe_text(entry.get("code")).lower()
    if code.endswith("_warning") or code == "warning":
        return True
    message = _safe_text(entry.get("message")).lower()
    return "warning" in message and "failed" not in message


def _diagnostic_counts(error_log: List[Dict[str, Any]]) -> tuple[int, int]:
    warning_count = 0
    error_count = 0
    for entry in error_log:
        if not isinstance(entry, Mapping):
            continue
        if _is_warning_diagnostic(entry):
            warning_count += 1
        else:
            error_count += 1
    return warning_count, error_count


def _resolve_workflow_status_from_exports(
    *,
    previous_status: str,
    requested_count: int,
    completed_count: int,
    failed_count: int,
) -> str:
    normalized_previous = _safe_text(previous_status).lower()
    if requested_count <= 0:
        return normalized_previous
    if failed_count == 0:
        if normalized_previous in {
            "partial_success",
            "degraded",
            "failed",
            "awaiting_clarification",
        }:
            return normalized_previous
        return normalized_previous or "success"
    if failed_count >= requested_count:
        if normalized_previous == "failed":
            return "failed"
        return "degraded"
    if normalized_previous in {"failed", "degraded"}:
        return normalized_previous
    return "partial_success"


def _filename_for_export(output_type: str, content: str, format_name: str) -> str:
    extension_map = {"markdown": "md", "html": "html", "pdf": "pdf", "docx": "docx"}
    normalized_format = str(format_name or "").strip().lower() or "markdown"
    extension = extension_map.get(normalized_format, "txt")
    digest = sha256(f"{output_type}:{content}".encode("utf-8")).hexdigest()[:12]
    return f"{output_type}_{digest}.{extension}"


def _build_export_outputs(
    assembled_outputs: Dict[str, Any],
    default_format: str,
) -> Dict[str, Dict[str, str]]:
    export_outputs: Dict[str, Dict[str, str]] = {}
    for output_type, raw_content in assembled_outputs.items():
        content = _safe_text(raw_content)
        if not content:
            continue
        format_name = default_format if default_format else "markdown"
        export_outputs[str(output_type)] = {
            "format": format_name,
            "content": content,
            "filename": _filename_for_export(str(output_type), content, format_name),
        }
    return export_outputs


def _safe_error_message(message: str, *, default: str) -> str:
    text = _safe_text(message)
    lowered = text.lower()
    if not text:
        return default
    if "traceback" in lowered or "stack trace" in lowered or '  file "' in lowered:
        return default
    if any(
        token in lowered
        for token in ("openai_api_key", "serp_api_key", "perplexity_api_key")
    ):
        return default
    return text


def _safe_error_entry(*, format_name: str, code: str, message: str) -> Dict[str, str]:
    return {
        "format": _safe_text(format_name).lower(),
        "code": _safe_text(code) or "export_error",
        "message": _safe_error_message(
            message,
            default="Export operation failed safely.",
        ),
    }


def _path_for_metadata(path_value: str) -> str:
    path_text = _safe_text(path_value)
    if not path_text:
        return ""
    try:
        resolved = Path(path_text).resolve()
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return path_text.replace("\\", "/")


def _is_safe_export_path(path_value: str, format_name: str) -> bool:
    path_text = _safe_text(path_value)
    fmt = _safe_text(format_name).lower()
    if not path_text or not fmt:
        return False
    expected_suffix = {
        "markdown": ".md",
        "html": ".html",
        "pdf": ".pdf",
        "docx": ".docx",
    }.get(fmt, "")
    if not expected_suffix:
        return False
    if not path_text.lower().endswith(expected_suffix):
        return False
    try:
        candidate = Path(path_text)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve()
        )
        export_dir = resolve_export_dir().resolve()
        return resolved.parent == export_dir
    except Exception:
        return False


def export_content(content: Any, format_name: str) -> Dict[str, Any]:
    """
    Deterministic export helper.

    Markdown/HTML/PDF write local files for download compatibility.
    Other formats remain deterministic path stubs for compatibility.
    """
    normalized = str(format_name or "").strip().lower()
    extension_map = {"markdown": "md", "html": "html", "pdf": "pdf", "docx": "docx"}
    if normalized not in extension_map:
        raise ValueError(f"Unsupported export format: {format_name}")

    if normalized == "markdown":
        resolved_path = resolve_markdown_export_path(content or "content")
        resolved_path.write_text(_safe_text(content), encoding="utf-8")
        return {"format": normalized, "path": _path_for_metadata(str(resolved_path))}

    if normalized == "html":
        resolved_path = resolve_html_export_path(content or "content")
        resolved_path.write_text(_safe_text(content), encoding="utf-8")
        return {"format": normalized, "path": _path_for_metadata(str(resolved_path))}

    if normalized == "pdf":
        if isinstance(content, bytes):
            pdf_bytes = content
            seed = sha256(content).hexdigest()
        else:
            text_content = _safe_text(content)
            seed = text_content or "content"
            pdf_bytes = build_pdf_document_bytes_from_text(text_content)
        resolved_path = resolve_pdf_export_path(seed or "content")
        resolved_path.write_bytes(pdf_bytes)
        return {"format": normalized, "path": _path_for_metadata(str(resolved_path))}

    if normalized == "docx":
        if isinstance(content, bytes):
            docx_bytes = content
            seed = sha256(content).hexdigest()
        else:
            text_content = _safe_text(content)
            seed = text_content or "content"
            docx_bytes = build_docx_document_bytes_from_text(text_content)
        resolved_path = resolve_docx_export_path(seed or "content")
        resolved_path.write_bytes(docx_bytes)
        return {"format": normalized, "path": _path_for_metadata(str(resolved_path))}

    if isinstance(content, bytes):
        digest_source = content
    else:
        digest_source = _safe_text(content).encode("utf-8")
    digest = sha256(digest_source).hexdigest()[:12]
    path = f"exports/content_{digest}.{extension_map[normalized]}"
    return {"format": normalized, "path": path.replace("\\", "/")}


def export_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Export final response and assembled outputs with safe markdown support."""
    final_response = str(state.get("final_response", ""))
    assembled_outputs = deepcopy(_safe_dict(state.get("assembled_outputs", {})))
    export_metadata = deepcopy(_safe_dict(state.get("export_metadata", {})))
    formats_requested = _normalize_requested_formats(
        _safe_list(export_metadata.get("formats_requested", []))
    )
    export_requested = bool(state.get("export_requested", False))
    if not formats_requested and export_requested:
        formats_requested = ["markdown"]

    export_paths: Dict[str, str] = {}
    error_log: List[Dict[str, Any]] = []
    export_status: Dict[str, str] = {}
    export_messages: List[str] = []
    format_validation: Dict[str, Dict[str, Any]] = {}

    default_export_format = formats_requested[0] if formats_requested else "markdown"
    export_outputs = _build_export_outputs(assembled_outputs, default_export_format)

    markdown_document = ""
    html_document = ""
    pdf_document = b""
    docx_document = b""
    markdown_validation = {"valid": True, "warnings": [], "errors": []}
    if any(fmt in {"markdown", "pdf", "docx"} for fmt in formats_requested):
        markdown_document = build_markdown_export_document(state)
        markdown_document = sanitize_markdown_content(markdown_document)
        markdown_validation = normalize_validation_result(
            validate_markdown_export(
                markdown_document,
                sources_exist=bool(_safe_list(state.get("sources", []))),
                sources=_safe_list(state.get("sources", [])),
            )
        )
        format_validation["markdown"] = markdown_validation
        for warning in markdown_validation["warnings"]:
            error_log.append(
                _safe_error_entry(
                    format_name="markdown",
                    code="markdown_validation_warning",
                    message=warning,
                )
            )
        if not markdown_validation["valid"]:
            for issue in markdown_validation["errors"]:
                error_log.append(
                    _safe_error_entry(
                        format_name="markdown",
                        code="markdown_validation_error",
                        message=issue,
                    )
                )

    if "html" in formats_requested:
        html_document = build_html_export_document(state)
        html_document = sanitize_html_content(html_document)
        html_validation = normalize_validation_result(
            validate_html_export(
                html_document,
                sources_exist=bool(_safe_list(state.get("sources", []))),
                sources=_safe_list(state.get("sources", [])),
            )
        )
        format_validation["html"] = html_validation
        for warning in html_validation["warnings"]:
            error_log.append(
                _safe_error_entry(
                    format_name="html",
                    code="html_validation_warning",
                    message=warning,
                )
            )
        if not html_validation["valid"]:
            for issue in html_validation["errors"]:
                error_log.append(
                    _safe_error_entry(
                        format_name="html",
                        code="html_validation_error",
                        message=issue,
                    )
                )

    if "pdf" in formats_requested:
        pdf_document = build_pdf_export_document(state)
        pdf_validation = normalize_validation_result(
            validate_pdf_export(
                pdf_document,
                sources_exist=bool(_safe_list(state.get("sources", []))),
                sources=_safe_list(state.get("sources", [])),
            )
        )
        format_validation["pdf"] = pdf_validation
        for warning in pdf_validation["warnings"]:
            error_log.append(
                _safe_error_entry(
                    format_name="pdf",
                    code="pdf_validation_warning",
                    message=warning,
                )
            )
        if not pdf_validation["valid"]:
            for issue in pdf_validation["errors"]:
                error_log.append(
                    _safe_error_entry(
                        format_name="pdf",
                        code="pdf_validation_error",
                        message=issue,
                    )
                )

    if "docx" in formats_requested:
        docx_document = build_docx_export_document(state)
        docx_validation = normalize_validation_result(
            validate_docx_export(
                docx_document,
                sources_exist=bool(_safe_list(state.get("sources", []))),
                sources=_safe_list(state.get("sources", [])),
            )
        )
        format_validation["docx"] = docx_validation
        for warning in docx_validation["warnings"]:
            error_log.append(
                _safe_error_entry(
                    format_name="docx",
                    code="docx_validation_warning",
                    message=warning,
                )
            )
        if not docx_validation["valid"]:
            for issue in docx_validation["errors"]:
                error_log.append(
                    _safe_error_entry(
                        format_name="docx",
                        code="docx_validation_error",
                        message=issue,
                    )
                )

    for fmt in formats_requested:
        if fmt not in _SUPPORTED_EXPORT_FORMATS:
            export_paths.pop(fmt, None)
            export_status[fmt] = "failed"
            export_messages.append(
                _safe_error_message(
                    f"{fmt.upper()} export format is not supported.",
                    default="Export validation failed safely.",
                )
            )
            error_log.append(
                _safe_error_entry(
                    format_name=fmt,
                    code="unsupported_export_format",
                    message=f"Unsupported export format: {fmt}",
                )
            )
            continue

        validation_result = _safe_dict(format_validation.get(fmt, {"valid": True}))
        if not bool(validation_result.get("valid", True)):
            export_paths.pop(fmt, None)
            export_status[fmt] = "failed"
            export_messages.append(
                _safe_error_message(
                    f"{fmt.upper()} export failed validation and was not delivered.",
                    default="Export validation failed safely.",
                )
            )
            error_log.append(
                _safe_error_entry(
                    format_name=fmt,
                    code=f"{fmt}_validation_failed",
                    message=(
                        f"{fmt.upper()} export failed validation and was "
                        "not delivered."
                    ),
                )
            )
            continue

        if fmt == "markdown":
            try:
                content_to_export = markdown_document or final_response
                result = export_content(content_to_export, "markdown")
                path = _safe_text(_safe_dict(result).get("path", ""))
                if path and _is_safe_export_path(path, "markdown"):
                    export_paths["markdown"] = _path_for_metadata(path)
                    export_status["markdown"] = "completed"
                else:
                    export_paths.pop("markdown", None)
                    export_status["markdown"] = "failed"
                    export_messages.append(
                        "Markdown export failed due to output-path validation."
                    )
                    error_log.append(
                        _safe_error_entry(
                            format_name="markdown",
                            code="markdown_export_path_invalid",
                            message=(
                                "Markdown export produced an unsafe or missing "
                                "output path."
                            ),
                        )
                    )
            except Exception:
                export_paths.pop("markdown", None)
                export_status["markdown"] = "failed"
                export_messages.append("Markdown export failed safely.")
                error_log.append(
                    _safe_error_entry(
                        format_name="markdown",
                        code="markdown_export_failed",
                        message="Markdown export failed safely.",
                    )
                )
            continue

        if fmt == "pdf":
            try:
                content_to_export = (
                    pdf_document
                    if pdf_document
                    else build_pdf_document_bytes_from_text(
                        markdown_document or final_response
                    )
                )
                result = export_content(content_to_export, "pdf")
                path = _safe_text(_safe_dict(result).get("path", ""))
                if path and _is_safe_export_path(path, "pdf"):
                    export_paths["pdf"] = _path_for_metadata(path)
                    export_status["pdf"] = "completed"
                else:
                    export_paths.pop("pdf", None)
                    export_status["pdf"] = "failed"
                    export_messages.append(
                        "PDF export failed due to output-path validation."
                    )
                    error_log.append(
                        _safe_error_entry(
                            format_name="pdf",
                            code="pdf_export_path_invalid",
                            message=(
                                "PDF export produced an unsafe or missing "
                                "output path."
                            ),
                        )
                    )
            except Exception:
                export_paths.pop("pdf", None)
                export_messages.append("PDF export failed safely.")
                error_log.append(
                    _safe_error_entry(
                        format_name="pdf",
                        code="pdf_export_failed",
                        message="PDF export failed safely.",
                    )
                )
                export_status["pdf"] = "failed"
            continue

        if fmt == "html":
            try:
                content_to_export = html_document or final_response
                result = export_content(content_to_export, "html")
                path = _safe_text(_safe_dict(result).get("path", ""))
                if path and _is_safe_export_path(path, "html"):
                    export_paths["html"] = _path_for_metadata(path)
                    export_status["html"] = "completed"
                else:
                    export_paths.pop("html", None)
                    export_status["html"] = "failed"
                    export_messages.append(
                        "HTML export failed due to output-path validation."
                    )
                    error_log.append(
                        _safe_error_entry(
                            format_name="html",
                            code="html_export_path_invalid",
                            message=(
                                "HTML export produced an unsafe or missing "
                                "output path."
                            ),
                        )
                    )
            except Exception:
                export_paths.pop("html", None)
                export_status["html"] = "failed"
                export_messages.append("HTML export failed safely.")
                error_log.append(
                    _safe_error_entry(
                        format_name="html",
                        code="html_export_failed",
                        message="HTML export failed safely.",
                    )
                )
            continue

        if fmt == "docx":
            try:
                content_to_export = (
                    docx_document
                    if docx_document
                    else build_docx_document_bytes_from_text(
                        markdown_document or final_response
                    )
                )
                result = export_content(content_to_export, "docx")
                path = _safe_text(_safe_dict(result).get("path", ""))
                if path and _is_safe_export_path(path, "docx"):
                    export_paths["docx"] = _path_for_metadata(path)
                    export_status["docx"] = "completed"
                else:
                    export_paths.pop("docx", None)
                    export_status["docx"] = "failed"
                    export_messages.append(
                        "DOCX export failed due to output-path validation."
                    )
                    error_log.append(
                        _safe_error_entry(
                            format_name="docx",
                            code="docx_export_path_invalid",
                            message=(
                                "DOCX export produced an unsafe or missing "
                                "output path."
                            ),
                        )
                    )
            except Exception:
                export_paths.pop("docx", None)
                export_status["docx"] = "failed"
                export_messages.append("DOCX export failed safely.")
                error_log.append(
                    _safe_error_entry(
                        format_name="docx",
                        code="docx_export_failed",
                        message="DOCX export failed safely.",
                    )
                )
            continue

        try:
            result = export_content(final_response, fmt)
            path = str(_safe_dict(result).get("path", "")).strip()
            if path and _is_safe_export_path(path, fmt):
                export_paths[fmt] = path
                export_status[fmt] = "completed"
            else:
                export_paths.pop(fmt, None)
                export_status[fmt] = "failed"
                export_messages.append(
                    _safe_error_message(
                        f"{fmt.upper()} export failed due to output-path validation.",
                        default="Export validation failed safely.",
                    )
                )
                error_log.append(
                    _safe_error_entry(
                        format_name=fmt,
                        code=f"{fmt}_export_path_invalid",
                        message=(
                            f"{fmt.upper()} export produced an unsafe or "
                            "missing output path."
                        ),
                    )
                )
        except Exception as exc:
            export_paths.pop(fmt, None)
            export_messages.append(
                _safe_error_message(
                    f"{fmt.upper()} export failed safely.",
                    default="Export validation failed safely.",
                )
            )
            error_log.append(
                _safe_error_entry(
                    format_name=fmt,
                    code=f"{fmt}_export_failed",
                    message=str(exc),
                )
            )
            export_status[fmt] = "failed"

    export_metadata["export_paths"] = export_paths
    export_metadata["error_log"] = error_log
    export_metadata["export_status"] = export_status
    requested_export_formats = list(formats_requested)
    completed_export_formats = [
        fmt
        for fmt in requested_export_formats
        if _safe_text(export_status.get(fmt)).lower() == "completed"
    ]
    failed_export_formats = [
        fmt
        for fmt in requested_export_formats
        if _safe_text(export_status.get(fmt)).lower() == "failed"
    ]
    warning_diagnostic_count, _ = _diagnostic_counts(error_log)
    export_error_count = len(failed_export_formats)

    export_metadata["requested_export_formats"] = requested_export_formats
    export_metadata["completed_export_formats"] = completed_export_formats
    export_metadata["failed_export_formats"] = failed_export_formats
    export_metadata["export_warning_count"] = warning_diagnostic_count
    export_metadata["export_error_count"] = export_error_count
    export_metadata["status_messages"] = list(
        dict.fromkeys(
            [
                _safe_error_message(
                    message,
                    default="Export validation failed safely.",
                )
                for message in export_messages
                if _safe_text(message)
            ]
        )
    )
    export_metadata["exported_at"] = _now_utc_iso()
    workflow_status = _resolve_workflow_status_from_exports(
        previous_status=_safe_text(state.get("workflow_status", "")),
        requested_count=len(requested_export_formats),
        completed_count=len(completed_export_formats),
        failed_count=export_error_count,
    )

    return {
        "export_metadata": export_metadata,
        "export_outputs": export_outputs,
        "workflow_status": workflow_status,
    }
