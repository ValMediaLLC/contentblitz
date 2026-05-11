"""Validation helpers for markdown export safety and structure."""

from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Dict, List, Mapping

_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
    "  file \"",
)
_RAW_PROVIDER_PAYLOAD_MARKERS = (
    "{'code':",
    '"code":',
    "configuration_error",
    "provider': 'openai'",
    '"provider": "openai"',
    "recoverable': false",
    '"recoverable": false',
)
_ENV_NAME_PATTERNS = (
    "openai_api_key",
    "serp_api_key",
    "perplexity_api_key",
)
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9\-_]{8,}\b"),
    re.compile(r"\bpplx-[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
    re.compile(r"\bserp_[A-Za-z0-9\-_]{8,}\b", flags=re.IGNORECASE),
)
_NONE_NULL_RE = re.compile(r"\b(?:none|null)\b", flags=re.IGNORECASE)
_SCRIPT_TAG_RE = re.compile(r"(?is)<script\b[^>]*>.*?</script>")
_UNSAFE_HTML_TAG_RE = re.compile(r"(?is)</?(?:iframe|object|embed)\b[^>]*>")
_EVENT_HANDLER_ATTR_RE = re.compile(
    r"""(?is)\s+on[a-z0-9_-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)"""
)
_JAVASCRIPT_URL_RE = re.compile(r"(?i)javascript:")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _contains_stack_trace(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STACK_TRACE_MARKERS)


def validate_markdown_export(
    markdown_text: str,
    *,
    sources_exist: bool = False,
) -> Dict[str, Any]:
    """
    Validate markdown export payload for structure and safety.

    Returns structured validation metadata:
    {"valid": bool, "warnings": [...], "errors": [...]}
    """
    text = _safe_text(markdown_text)
    warnings: List[str] = []
    errors: List[str] = []

    if not text:
        errors.append("Export content is empty.")
        return {"valid": False, "warnings": warnings, "errors": errors}

    lines = text.splitlines()
    first_non_empty = next((line.strip() for line in lines if line.strip()), "")
    if not first_non_empty.startswith("# "):
        errors.append("Missing required top-level markdown heading.")

    lowered = text.lower()
    if _NONE_NULL_RE.search(text):
        warnings.append("Found null-like placeholder text.")
    if _contains_stack_trace(text):
        errors.append("Stack trace content is not allowed in exports.")
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        errors.append("Base64 image payload is not allowed in exports.")
    if any(env_name in lowered for env_name in _ENV_NAME_PATTERNS):
        errors.append("Environment variable names are not allowed in exports.")
    if any(pattern.search(text) for pattern in _TOKEN_PATTERNS):
        errors.append("Credential-like token content is not allowed in exports.")
    if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
        errors.append("Raw provider/configuration payload content is not allowed in exports.")
    if sources_exist and "## sources" not in lowered:
        errors.append("Sources section is required when sources are present.")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def validate_html_export(
    html_text: str,
    *,
    sources_exist: bool = False,
) -> Dict[str, Any]:
    """
    Validate html export payload for structure and safety.

    Returns structured validation metadata:
    {"valid": bool, "warnings": [...], "errors": [...]}
    """
    text = _safe_text(html_text)
    warnings: List[str] = []
    errors: List[str] = []

    if not text:
        errors.append("Export content is empty.")
        return {"valid": False, "warnings": warnings, "errors": errors}

    lowered = text.lower()
    if "<!doctype html>" not in lowered:
        errors.append("Missing html doctype.")
    if "<html" not in lowered or "</html>" not in lowered:
        errors.append("Malformed html document wrapper.")
    if "<body" not in lowered or "</body>" not in lowered:
        errors.append("Malformed html body wrapper.")

    if _NONE_NULL_RE.search(text):
        warnings.append("Found null-like placeholder text.")
    if _contains_stack_trace(text):
        errors.append("Stack trace content is not allowed in exports.")
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        errors.append("Base64 image payload is not allowed in exports.")
    if any(env_name in lowered for env_name in _ENV_NAME_PATTERNS):
        errors.append("Environment variable names are not allowed in exports.")
    if any(pattern.search(text) for pattern in _TOKEN_PATTERNS):
        errors.append("Credential-like token content is not allowed in exports.")
    if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
        errors.append("Raw provider/configuration payload content is not allowed in exports.")
    if _SCRIPT_TAG_RE.search(text):
        errors.append("Script tags are not allowed in exports.")
    if _UNSAFE_HTML_TAG_RE.search(text):
        errors.append("Unsafe embed tags are not allowed in exports.")
    if _EVENT_HANDLER_ATTR_RE.search(text):
        errors.append("Inline javascript handlers are not allowed in exports.")
    if _JAVASCRIPT_URL_RE.search(text):
        errors.append("javascript: URLs are not allowed in exports.")
    if sources_exist and "<h2>sources</h2>" not in lowered:
        errors.append("Sources section is required when sources are present.")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def validate_pdf_export(
    pdf_payload: bytes | str,
    *,
    sources_exist: bool = False,
) -> Dict[str, Any]:
    """
    Validate pdf export payload for structure and safety.

    Returns structured validation metadata:
    {"valid": bool, "warnings": [...], "errors": [...]}
    """
    if isinstance(pdf_payload, bytes):
        binary = pdf_payload
        text = pdf_payload.decode("latin-1", errors="ignore")
    else:
        text = _safe_text(pdf_payload)
        binary = text.encode("latin-1", errors="ignore")

    warnings: List[str] = []
    errors: List[str] = []

    if not binary:
        errors.append("Export content is empty.")
        return {"valid": False, "warnings": warnings, "errors": errors}

    if not binary.startswith(b"%PDF-"):
        errors.append("Missing pdf header.")
    if b"xref" not in binary:
        errors.append("Missing pdf cross-reference table.")
    if b"trailer" not in binary:
        errors.append("Missing pdf trailer.")
    if b"%%EOF" not in binary:
        errors.append("Missing pdf EOF marker.")

    lowered = text.lower()
    if _NONE_NULL_RE.search(text):
        warnings.append("Found null-like placeholder text.")
    if _contains_stack_trace(text):
        errors.append("Stack trace content is not allowed in exports.")
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        errors.append("Base64 image payload is not allowed in exports.")
    if any(env_name in lowered for env_name in _ENV_NAME_PATTERNS):
        errors.append("Environment variable names are not allowed in exports.")
    if any(pattern.search(text) for pattern in _TOKEN_PATTERNS):
        errors.append("Credential-like token content is not allowed in exports.")
    if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
        errors.append("Raw provider/configuration payload content is not allowed in exports.")
    if _SCRIPT_TAG_RE.search(text):
        errors.append("Script tags are not allowed in exports.")
    if _UNSAFE_HTML_TAG_RE.search(text):
        errors.append("Unsafe embed tags are not allowed in exports.")
    if _EVENT_HANDLER_ATTR_RE.search(text):
        errors.append("Inline javascript handlers are not allowed in exports.")
    if _JAVASCRIPT_URL_RE.search(text):
        errors.append("javascript: URLs are not allowed in exports.")
    if _CONTROL_CHARS_RE.search(text):
        warnings.append("Found control characters in export content.")
    if sources_exist and "sources" not in lowered:
        errors.append("Sources section is required when sources are present.")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def validate_docx_export(
    docx_payload: bytes | str,
    *,
    sources_exist: bool = False,
) -> Dict[str, Any]:
    """
    Validate docx export payload for structure and safety.

    Returns structured validation metadata:
    {"valid": bool, "warnings": [...], "errors": [...]}
    """
    if isinstance(docx_payload, bytes):
        binary = docx_payload
    else:
        text = _safe_text(docx_payload)
        binary = text.encode("utf-8", errors="ignore")

    warnings: List[str] = []
    errors: List[str] = []

    if not binary:
        errors.append("Export content is empty.")
        return {"valid": False, "warnings": warnings, "errors": errors}

    if not binary.startswith(b"PK"):
        errors.append("Missing docx zip signature.")
        return {"valid": False, "warnings": warnings, "errors": errors}

    try:
        with zipfile.ZipFile(io.BytesIO(binary), mode="r") as archive:
            names = set(archive.namelist())
            required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml"}
            missing = sorted(required.difference(names))
            if missing:
                errors.append("Missing required docx parts: " + ", ".join(missing))
                return {"valid": False, "warnings": warnings, "errors": errors}

            content_types_xml = archive.read("[Content_Types].xml").decode(
                "utf-8",
                errors="ignore",
            )
            document_xml = archive.read("word/document.xml").decode(
                "utf-8",
                errors="ignore",
            )
    except zipfile.BadZipFile:
        errors.append("Malformed docx archive.")
        return {"valid": False, "warnings": warnings, "errors": errors}
    except KeyError:
        errors.append("Required docx content is missing.")
        return {"valid": False, "warnings": warnings, "errors": errors}

    if "word/document.xml" not in content_types_xml:
        errors.append("DOCX content types missing word/document.xml entry.")

    lowered = document_xml.lower()
    if _NONE_NULL_RE.search(document_xml):
        warnings.append("Found null-like placeholder text.")
    if _contains_stack_trace(document_xml):
        errors.append("Stack trace content is not allowed in exports.")
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        errors.append("Base64 image payload is not allowed in exports.")
    if any(env_name in lowered for env_name in _ENV_NAME_PATTERNS):
        errors.append("Environment variable names are not allowed in exports.")
    if any(pattern.search(document_xml) for pattern in _TOKEN_PATTERNS):
        errors.append("Credential-like token content is not allowed in exports.")
    if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
        errors.append("Raw provider/configuration payload content is not allowed in exports.")
    if _SCRIPT_TAG_RE.search(document_xml):
        errors.append("Script tags are not allowed in exports.")
    if _UNSAFE_HTML_TAG_RE.search(document_xml):
        errors.append("Unsafe embed tags are not allowed in exports.")
    if _EVENT_HANDLER_ATTR_RE.search(document_xml):
        errors.append("Inline javascript handlers are not allowed in exports.")
    if _JAVASCRIPT_URL_RE.search(document_xml):
        errors.append("javascript: URLs are not allowed in exports.")
    if _CONTROL_CHARS_RE.search(document_xml):
        warnings.append("Found control characters in export content.")
    if sources_exist and "sources" not in lowered:
        errors.append("Sources section is required when sources are present.")

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def normalize_validation_result(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize arbitrary validation payload into a strict safe shape."""
    valid = bool(payload.get("valid", False))
    warnings = [
        _safe_text(item) for item in payload.get("warnings", []) if _safe_text(item)
    ] if isinstance(payload.get("warnings", []), list) else []
    errors = [
        _safe_text(item) for item in payload.get("errors", []) if _safe_text(item)
    ] if isinstance(payload.get("errors", []), list) else []
    return {"valid": valid, "warnings": warnings, "errors": errors}
