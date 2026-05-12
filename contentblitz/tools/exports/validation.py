"""Validation helpers for markdown export safety and structure."""

from __future__ import annotations

import io
import re
from urllib.parse import urlsplit
import zipfile
from typing import Any, Dict, List, Mapping

from contentblitz.quality.citations import (
    CITATION_VALIDATION_WARNING,
    validate_citation_sources,
)

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
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HTML_HREF_RE = re.compile(
    r"""(?is)\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')"""
)
_MARKDOWN_SOURCE_ENTRY_RE = re.compile(
    r"^(?:\d+\.\s*|\[\d+\]\s+|[-*]\s+)(.+)$"
)
_MARKDOWN_SECTION_RE = re.compile(r"(?im)^##\s+([^\n]+)\s*$")
_MIN_TEXT_EXPORT_LENGTH = 60
_MIN_BINARY_EXPORT_LENGTH = 200


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _contains_stack_trace(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STACK_TRACE_MARKERS)


def _is_safe_url(url: str) -> bool:
    candidate = _safe_text(url)
    if not candidate:
        return False
    lowered = candidate.lower()
    if lowered.startswith(("javascript:", "data:", "file:")):
        return False
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False
    if any(ch in candidate for ch in ("\r", "\n", "\t")):
        return False
    return True


def _extract_markdown_section(markdown_text: str, section_name: str) -> str:
    lines = markdown_text.splitlines()
    start = -1
    target = section_name.strip().lower()
    for index, line in enumerate(lines):
        match = _MARKDOWN_SECTION_RE.match(line)
        if not match:
            continue
        if match.group(1).strip().lower() == target:
            start = index + 1
            break
    if start < 0:
        return ""
    section_lines: list[str] = []
    for line in lines[start:]:
        if _MARKDOWN_SECTION_RE.match(line):
            break
        section_lines.append(line)
    return "\n".join(section_lines).strip()


def _markdown_invalid_urls(markdown_text: str) -> list[str]:
    invalid: list[str] = []
    for _label, url in _MARKDOWN_LINK_RE.findall(markdown_text):
        if not _is_safe_url(url):
            invalid.append(_safe_text(url))
    return invalid


def _html_invalid_urls(html_text: str) -> list[str]:
    invalid: list[str] = []
    for match in _HTML_HREF_RE.findall(html_text):
        url = _safe_text(match[0] or match[1])
        if not url:
            continue
        if not _is_safe_url(url):
            invalid.append(url)
    return invalid


def _apply_citation_warning(
    warnings: list[str],
    *,
    sources_exist: bool,
    sources: Any,
) -> None:
    citation_result = validate_citation_sources(
        sources,
        research_requested=sources_exist,
    )
    if str(citation_result.get("status", "")).lower() == "degraded":
        if CITATION_VALIDATION_WARNING not in warnings:
            warnings.append(CITATION_VALIDATION_WARNING)


def validate_markdown_export(
    markdown_text: str,
    *,
    sources_exist: bool = False,
    sources: Any | None = None,
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
    if len(text) < _MIN_TEXT_EXPORT_LENGTH:
        errors.append("Export content is too short to be deliverable.")

    lines = text.splitlines()
    first_non_empty = next((line.strip() for line in lines if line.strip()), "")
    if first_non_empty != "# ContentBlitz Export":
        errors.append("Missing required top-level markdown heading.")
    if "## Workflow Summary" not in text:
        errors.append("Missing required Workflow Summary section.")
    if "Workflow Status:" not in text:
        errors.append("Missing required workflow summary fields.")

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
    invalid_urls = _markdown_invalid_urls(text)
    if invalid_urls:
        errors.append("Unsafe or invalid URL content is not allowed in exports.")
    sources_section = _extract_markdown_section(text, "Sources")
    if sources_exist and "## sources" not in lowered:
        errors.append("Sources section is required when sources are present.")
    if sources_exist and not sources_section:
        errors.append("Sources section is required when sources are present.")
    if sources_section:
        source_entries: list[str] = []
        for line in sources_section.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            entry_match = _MARKDOWN_SOURCE_ENTRY_RE.match(stripped)
            if not entry_match:
                continue
            entry = _safe_text(entry_match.group(1))
            if not entry:
                continue
            source_entries.append(entry)
            if any(marker in entry.lower() for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
                errors.append("Invalid source/citation entry detected.")
        if sources_exist and not source_entries:
            errors.append("Sources section must include readable citations.")
        for entry in source_entries:
            cleaned = _MARKDOWN_LINK_RE.sub(r"\1", entry).strip()
            if not cleaned:
                errors.append("Sources section must include readable citations.")
                break

    _apply_citation_warning(
        warnings,
        sources_exist=sources_exist,
        sources=sources if sources is not None else [],
    )

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def validate_html_export(
    html_text: str,
    *,
    sources_exist: bool = False,
    sources: Any | None = None,
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
    if len(text) < _MIN_TEXT_EXPORT_LENGTH:
        errors.append("Export content is too short to be deliverable.")

    lowered = text.lower()
    if "<!doctype html>" not in lowered:
        errors.append("Missing html doctype.")
    if "<html" not in lowered or "</html>" not in lowered:
        errors.append("Malformed html document wrapper.")
    if "<body" not in lowered or "</body>" not in lowered:
        errors.append("Malformed html body wrapper.")
    if "<h1>contentblitz export</h1>" not in lowered:
        errors.append("Missing required export title.")
    if "<h2>workflow summary</h2>" not in lowered:
        errors.append("Missing required Workflow Summary section.")

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
    invalid_urls = _html_invalid_urls(text)
    if invalid_urls:
        errors.append("Unsafe or invalid URL content is not allowed in exports.")
    if sources_exist and "<h2>sources</h2>" not in lowered:
        errors.append("Sources section is required when sources are present.")
    if sources_exist and "<h2>sources</h2>" in lowered:
        sources_tail = lowered.split("<h2>sources</h2>", 1)[1]
        if "<li" not in sources_tail:
            errors.append("Sources section must include readable citations.")

    _apply_citation_warning(
        warnings,
        sources_exist=sources_exist,
        sources=sources if sources is not None else [],
    )

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def validate_pdf_export(
    pdf_payload: bytes | str,
    *,
    sources_exist: bool = False,
    sources: Any | None = None,
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
    if len(binary) < _MIN_BINARY_EXPORT_LENGTH:
        errors.append("Export file is too small to be deliverable.")

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
    if "contentblitz export" not in lowered:
        errors.append("Missing required export title.")
    if "workflow summary" not in lowered:
        errors.append("Missing required Workflow Summary section.")
    if sources_exist and "sources" not in lowered:
        errors.append("Sources section is required when sources are present.")

    _apply_citation_warning(
        warnings,
        sources_exist=sources_exist,
        sources=sources if sources is not None else [],
    )

    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
    }


def validate_docx_export(
    docx_payload: bytes | str,
    *,
    sources_exist: bool = False,
    sources: Any | None = None,
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
    if len(binary) < _MIN_BINARY_EXPORT_LENGTH:
        errors.append("Export file is too small to be deliverable.")
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
    if "contentblitz export" not in lowered:
        errors.append("Missing required export title.")
    if "workflow summary" not in lowered:
        errors.append("Missing required Workflow Summary section.")
    if sources_exist and "sources" not in lowered:
        errors.append("Sources section is required when sources are present.")

    _apply_citation_warning(
        warnings,
        sources_exist=sources_exist,
        sources=sources if sources is not None else [],
    )

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
