"""Export node implementation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

from contentblitz.tools.exports.filenames import (
    resolve_html_export_path,
    resolve_markdown_export_path,
)
from contentblitz.tools.exports.html import (
    build_html_export_document,
    sanitize_html_content,
)
from contentblitz.tools.exports.markdown import (
    build_markdown_export_document,
    sanitize_markdown_content,
)
from contentblitz.tools.exports.validation import (
    normalize_validation_result,
    validate_html_export,
    validate_markdown_export,
)

_SUPPORTED_EXPORT_FORMATS = {"markdown", "html", "pdf"}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _filename_for_export(output_type: str, content: str, format_name: str) -> str:
    extension_map = {"markdown": "md", "html": "html", "pdf": "pdf"}
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
    if "traceback" in lowered or "stack trace" in lowered or "  file \"" in lowered:
        return default
    if any(token in lowered for token in ("openai_api_key", "serp_api_key", "perplexity_api_key")):
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


def export_content(content: str, format_name: str) -> Dict[str, Any]:
    """
    Deterministic export helper.

    Markdown/HTML write local files for download compatibility.
    Other formats remain deterministic path stubs for compatibility.
    """
    normalized = str(format_name or "").strip().lower()
    extension_map = {"markdown": "md", "html": "html", "pdf": "pdf"}
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

    digest = sha256((content or "").encode("utf-8")).hexdigest()[:12]
    path = f"exports/content_{digest}.{extension_map[normalized]}"
    return {"format": normalized, "path": path.replace("\\", "/")}


def export_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Export final response and assembled outputs with safe markdown support."""
    final_response = str(state.get("final_response", ""))
    assembled_outputs = deepcopy(_safe_dict(state.get("assembled_outputs", {})))
    export_metadata = deepcopy(_safe_dict(state.get("export_metadata", {})))
    formats_requested = [
        str(item).strip().lower()
        for item in _safe_list(export_metadata.get("formats_requested", []))
        if str(item).strip()
    ]
    export_requested = bool(state.get("export_requested", False))
    if not formats_requested and export_requested:
        formats_requested = ["markdown"]

    export_paths = deepcopy(_safe_dict(export_metadata.get("export_paths", {})))
    error_log = deepcopy(_safe_list(export_metadata.get("error_log", [])))
    export_status = deepcopy(_safe_dict(export_metadata.get("export_status", {})))

    default_export_format = formats_requested[0] if formats_requested else "markdown"
    export_outputs = _build_export_outputs(assembled_outputs, default_export_format)

    markdown_document = ""
    html_document = ""
    markdown_validation = {"valid": True, "warnings": [], "errors": []}
    if any(fmt in {"markdown", "pdf"} for fmt in formats_requested):
        markdown_document = build_markdown_export_document(state)
        markdown_document = sanitize_markdown_content(markdown_document)
        markdown_validation = normalize_validation_result(
            validate_markdown_export(
                markdown_document,
                sources_exist=bool(_safe_list(state.get("sources", []))),
            )
        )
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
            )
        )
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

    for fmt in formats_requested:
        if fmt not in _SUPPORTED_EXPORT_FORMATS:
            export_status[fmt] = "failed"
            error_log.append(
                _safe_error_entry(
                    format_name=fmt,
                    code="unsupported_export_format",
                    message=f"Unsupported export format: {fmt}",
                )
            )
            continue

        if fmt == "markdown":
            try:
                content_to_export = markdown_document or final_response
                result = export_content(content_to_export, "markdown")
                path = _safe_text(_safe_dict(result).get("path", ""))
                if path:
                    export_paths["markdown"] = _path_for_metadata(path)
                    export_status["markdown"] = "completed"
                else:
                    export_status["markdown"] = "failed"
                    error_log.append(
                        _safe_error_entry(
                            format_name="markdown",
                            code="markdown_export_path_missing",
                            message="Markdown export returned no output path.",
                        )
                    )
            except Exception:
                export_status["markdown"] = "failed"
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
                result = export_content(markdown_document or final_response, "pdf")
                export_paths["pdf"] = str(_safe_dict(result).get("path", "")).strip()
                export_status["pdf"] = "completed"
            except Exception as exc:
                error_log.append(
                    _safe_error_entry(
                        format_name="pdf",
                        code="pdf_export_failed",
                        message=str(exc),
                    )
                )
                export_status["pdf"] = "failed"
                try:
                    fallback = export_content(markdown_document or final_response, "markdown")
                    fallback_path = str(_safe_dict(fallback).get("path", "")).strip()
                    if fallback_path:
                        export_paths["pdf"] = fallback_path
                        export_paths.setdefault("markdown", fallback_path)
                        export_status.setdefault("markdown", "completed")
                except Exception as fallback_exc:
                    error_log.append(
                        _safe_error_entry(
                            format_name="markdown",
                            code="pdf_markdown_fallback_failed",
                            message=f"PDF fallback failed: {fallback_exc}",
                        )
                    )
            continue

        if fmt == "html":
            try:
                content_to_export = html_document or final_response
                result = export_content(content_to_export, "html")
                path = _safe_text(_safe_dict(result).get("path", ""))
                if path:
                    export_paths["html"] = _path_for_metadata(path)
                    export_status["html"] = "completed"
                else:
                    export_status["html"] = "failed"
                    error_log.append(
                        _safe_error_entry(
                            format_name="html",
                            code="html_export_path_missing",
                            message="HTML export returned no output path.",
                        )
                    )
            except Exception:
                export_status["html"] = "failed"
                error_log.append(
                    _safe_error_entry(
                        format_name="html",
                        code="html_export_failed",
                        message="HTML export failed safely.",
                    )
                )
            continue

        try:
            result = export_content(final_response, fmt)
            path = str(_safe_dict(result).get("path", "")).strip()
            if path:
                export_paths[fmt] = path
                export_status[fmt] = "completed"
        except Exception as exc:
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
    export_metadata["exported_at"] = _now_utc_iso()

    return {
        "export_metadata": export_metadata,
        "export_outputs": export_outputs,
    }
