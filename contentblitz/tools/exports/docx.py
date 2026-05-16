"""DOCX export renderer and sanitizer for ContentBlitz."""

from __future__ import annotations

import io
import re
import zipfile
from typing import Any, List, Mapping
from xml.sax.saxutils import escape

from contentblitz.safety.output_sanitizer import sanitize_plain_output
from contentblitz.tools.exports.markdown import build_markdown_export_document

_ENV_NAME_RE = re.compile(
    r"OPENAI_API_KEY|SERP_API_KEY|PERPLEXITY_API_KEY",
    flags=re.IGNORECASE,
)
_TOKEN_RE = re.compile(
    r"\b(?:sk|pplx|serp)_[A-Za-z0-9\-_]{8,}\b|\bsk-[A-Za-z0-9\-_]{8,}\b|\bpplx-[A-Za-z0-9\-_]{8,}\b",
    flags=re.IGNORECASE,
)
_NONE_NULL_RE = re.compile(r"\b(?:none|null)\b", flags=re.IGNORECASE)
_STACK_TRACE_MARKERS = (
    "traceback (most recent call last):",
    "stack trace",
    '  file "',
)
_RAW_PROVIDER_PAYLOAD_MARKERS = (
    "{'code':",
    '"code":',
    "configuration_error",
    "provider':",
    '"provider":',
    "recoverable': false",
    '"recoverable": false',
)
_SCRIPT_TAG_RE = re.compile(r"(?is)<script\b[^>]*>.*?</script>")
_UNSAFE_TAG_RE = re.compile(r"(?is)</?(?:iframe|object|embed)\b[^>]*>")
_HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
_EVENT_HANDLER_ATTR_RE = re.compile(
    r"""(?is)\s+on[a-z0-9_-]+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)"""
)
_JAVASCRIPT_URL_RE = re.compile(r"(?i)javascript:")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ORDERED_LIST_RE = re.compile(r"^\d+\.\s+")
_UNORDERED_LIST_RE = re.compile(r"^[-*]\s+")
_GENERIC_RECOVERABLE_WARNING = "A recoverable workflow issue was encountered."
_WORD_MAIN_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _sanitize_plain_text(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        return ""

    lowered = raw.lower()
    if any(marker in lowered for marker in _STACK_TRACE_MARKERS):
        return ""
    if "data:image/" in lowered or "base64" in lowered or "b64_json" in lowered:
        return ""
    if any(marker in lowered for marker in _RAW_PROVIDER_PAYLOAD_MARKERS):
        return _GENERIC_RECOVERABLE_WARNING

    clean = _SCRIPT_TAG_RE.sub("", raw)
    clean = _UNSAFE_TAG_RE.sub("", clean)
    clean = _EVENT_HANDLER_ATTR_RE.sub("", clean)
    clean = _JAVASCRIPT_URL_RE.sub("", clean)
    clean = _HTML_TAG_RE.sub("", clean)
    clean = _CONTROL_CHARS_RE.sub("", clean)
    clean = _MARKDOWN_LINK_RE.sub(r"\1 (\2)", clean)
    clean = clean.replace("`", "").replace("**", "").replace("__", "")
    clean = _ENV_NAME_RE.sub("[REDACTED]", clean)
    clean = _TOKEN_RE.sub("[REDACTED]", clean)
    clean = _NONE_NULL_RE.sub("", clean)
    sanitized, _ = sanitize_plain_output(clean)
    return sanitized.strip()


def _markdown_to_lines(markdown_text: str) -> List[str]:
    raw_lines = markdown_text.splitlines()
    lines: List[str] = []
    previous_blank = False
    for raw in raw_lines:
        stripped = raw.rstrip()
        if not stripped:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(stripped)
        previous_blank = False
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _build_docx_paragraph_xml(
    *, text: str, heading_level: int | None, list_kind: str | None
) -> str:
    xml_space = (
        ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    )
    escaped_text = escape(text)

    ppr_segments: List[str] = []
    run_props_segments: List[str] = []

    if heading_level:
        run_props_segments.append("<w:b/>")
        heading_size = {1: "40", 2: "32", 3: "28"}.get(heading_level, "24")
        run_props_segments.append(f'<w:sz w:val="{heading_size}"/>')
        ppr_segments.append('<w:spacing w:before="240" w:after="120"/>')
    elif list_kind:
        ppr_segments.append('<w:spacing w:after="60"/>')
    else:
        ppr_segments.append('<w:spacing w:after="100"/>')

    ppr = f"<w:pPr>{''.join(ppr_segments)}</w:pPr>" if ppr_segments else ""
    run_props = (
        f"<w:rPr>{''.join(run_props_segments)}</w:rPr>" if run_props_segments else ""
    )
    return f"<w:p>{ppr}<w:r>{run_props}<w:t{xml_space}>{escaped_text}</w:t></w:r></w:p>"


def _build_document_xml_from_lines(lines: List[str]) -> str:
    body_parts: List[str] = []

    for line in lines:
        if not line:
            body_parts.append("<w:p/>")
            continue

        heading_match = _MARKDOWN_HEADING_RE.match(line)
        if heading_match:
            heading_level = min(len(heading_match.group(1)), 3)
            heading_text = _sanitize_plain_text(heading_match.group(2))
            if heading_text:
                body_parts.append(
                    _build_docx_paragraph_xml(
                        text=heading_text,
                        heading_level=heading_level,
                        list_kind=None,
                    )
                )
            continue

        list_kind: str | None = None
        content = line
        if _UNORDERED_LIST_RE.match(line):
            list_kind = "unordered"
            content = f"- {_UNORDERED_LIST_RE.sub('', line, count=1).strip()}"
        elif _ORDERED_LIST_RE.match(line):
            list_kind = "ordered"
            content = line

        paragraph_text = _sanitize_plain_text(content)
        if not paragraph_text:
            continue
        body_parts.append(
            _build_docx_paragraph_xml(
                text=paragraph_text,
                heading_level=None,
                list_kind=list_kind,
            )
        )

    if not body_parts:
        body_parts.append(
            _build_docx_paragraph_xml(
                text="ContentBlitz Export",
                heading_level=1,
                list_kind=None,
            )
        )

    sect_pr = (
        "<w:sectPr>"
        '<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"'
        ' w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    body_xml = "".join(body_parts) + sect_pr
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_WORD_MAIN_NS}">'
        f"<w:body>{body_xml}</w:body>"
        "</w:document>"
    )


def _docx_content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )


def _docx_root_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )


def _writestr_deterministic(
    archive: zipfile.ZipFile,
    *,
    name: str,
    data: str,
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    archive.writestr(info, data.encode("utf-8"))


def build_docx_document_bytes_from_text(text: str) -> bytes:
    """Build deterministic DOCX bytes from already-sanitized export text."""
    normalized = _sanitize_plain_text(text)
    lines = _markdown_to_lines(normalized or "ContentBlitz Export")
    document_xml = _build_document_xml_from_lines(lines)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        _writestr_deterministic(
            archive,
            name="[Content_Types].xml",
            data=_docx_content_types_xml(),
        )
        _writestr_deterministic(
            archive,
            name="_rels/.rels",
            data=_docx_root_relationships_xml(),
        )
        _writestr_deterministic(
            archive,
            name="word/document.xml",
            data=document_xml,
        )
    return buffer.getvalue()


def build_docx_export_document(state: Mapping[str, Any]) -> bytes:
    """Build deterministic DOCX bytes from workflow state."""
    markdown_document = build_markdown_export_document(state)
    lines = _markdown_to_lines(markdown_document)
    sanitized_lines = [_sanitize_plain_text(line) if line else "" for line in lines]
    normalized_lines: List[str] = []
    previous_blank = False
    for line in sanitized_lines:
        if not line:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False
    document_xml = _build_document_xml_from_lines(normalized_lines)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        _writestr_deterministic(
            archive,
            name="[Content_Types].xml",
            data=_docx_content_types_xml(),
        )
        _writestr_deterministic(
            archive,
            name="_rels/.rels",
            data=_docx_root_relationships_xml(),
        )
        _writestr_deterministic(
            archive,
            name="word/document.xml",
            data=document_xml,
        )
    return buffer.getvalue()


def docx_mime_type() -> str:
    """Return DOCX MIME type used by metadata validation tests."""
    return _DOCX_MIME
