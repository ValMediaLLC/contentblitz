"""Export node implementation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List


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


def export_content(content: str, format_name: str) -> Dict[str, Any]:
    """
    Deterministic export stub.

    This function is intentionally mockable in tests and performs no real I/O.
    """
    normalized = str(format_name or "").strip().lower()
    extension_map = {"markdown": "md", "html": "html", "pdf": "pdf"}
    if normalized not in extension_map:
        raise ValueError(f"Unsupported export format: {format_name}")

    digest = sha256((content or "").encode("utf-8")).hexdigest()[:12]
    path = f"exports/content_{digest}.{extension_map[normalized]}"
    return {"format": normalized, "path": path}


def export_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Export final_response to requested formats with deterministic fallback behavior."""
    final_response = str(state.get("final_response", ""))
    assembled_outputs = deepcopy(_safe_dict(state.get("assembled_outputs", {})))
    export_metadata = deepcopy(_safe_dict(state.get("export_metadata", {})))
    formats_requested = [
        str(item).strip().lower()
        for item in _safe_list(export_metadata.get("formats_requested", []))
        if str(item).strip()
    ]

    export_paths = deepcopy(_safe_dict(export_metadata.get("export_paths", {})))
    error_log = deepcopy(_safe_list(export_metadata.get("error_log", [])))
    default_export_format = formats_requested[0] if formats_requested else "markdown"
    export_outputs = _build_export_outputs(assembled_outputs, default_export_format)

    for fmt in formats_requested:
        if fmt == "pdf":
            try:
                result = export_content(final_response, "pdf")
                export_paths["pdf"] = str(_safe_dict(result).get("path", "")).strip()
            except Exception as exc:
                error_log.append(
                    {
                        "format": "pdf",
                        "message": str(exc),
                    }
                )
                try:
                    fallback = export_content(final_response, "markdown")
                    fallback_path = str(_safe_dict(fallback).get("path", "")).strip()
                    if fallback_path:
                        export_paths["pdf"] = fallback_path
                        export_paths.setdefault("markdown", fallback_path)
                except Exception as fallback_exc:
                    error_log.append(
                        {
                            "format": "markdown",
                            "message": f"PDF fallback failed: {fallback_exc}",
                        }
                    )
            continue

        try:
            result = export_content(final_response, fmt)
            path = str(_safe_dict(result).get("path", "")).strip()
            if path:
                export_paths[fmt] = path
        except Exception as exc:
            error_log.append(
                {
                    "format": fmt,
                    "message": str(exc),
                }
            )

    export_metadata["export_paths"] = export_paths
    export_metadata["error_log"] = error_log
    export_metadata["exported_at"] = _now_utc_iso()

    return {
        "export_metadata": export_metadata,
        "export_outputs": export_outputs,
    }
