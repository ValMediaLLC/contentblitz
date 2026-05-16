"""PDF export renderer and sanitizer for ContentBlitz."""

from __future__ import annotations

import re
import textwrap
from typing import Any, List, Mapping

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
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_ORDERED_LIST_RE = re.compile(r"^\d+\.\s+")
_UNORDERED_LIST_RE = re.compile(r"^[-*]\s+")
_GENERIC_RECOVERABLE_WARNING = "A recoverable workflow issue was encountered."
_MAX_LINE_WIDTH = 96
_MAX_LINES_PER_PAGE = 44


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
    clean = _ENV_NAME_RE.sub("[REDACTED]", clean)
    clean = _TOKEN_RE.sub("[REDACTED]", clean)
    clean = _NONE_NULL_RE.sub("", clean)
    sanitized, _ = sanitize_plain_output(clean)
    return sanitized.strip()


def _normalize_markdown_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    stripped = _MARKDOWN_HEADING_RE.sub("", stripped)
    stripped = _MARKDOWN_LINK_RE.sub(r"\1 (\2)", stripped)
    if _UNORDERED_LIST_RE.match(stripped):
        body = _UNORDERED_LIST_RE.sub("", stripped, count=1)
        return f"- {body.strip()}"
    if _ORDERED_LIST_RE.match(stripped):
        return stripped
    return stripped


def _build_pdf_export_text(state: Mapping[str, Any]) -> str:
    markdown_document = build_markdown_export_document(state)
    raw_lines = markdown_document.splitlines()
    sanitized_lines: List[str] = []
    previous_blank = False
    for raw in raw_lines:
        normalized = _normalize_markdown_line(raw)
        cleaned = _sanitize_plain_text(normalized)
        if not cleaned:
            if previous_blank:
                continue
            sanitized_lines.append("")
            previous_blank = True
            continue
        sanitized_lines.append(cleaned)
        previous_blank = False

    while sanitized_lines and not sanitized_lines[0]:
        sanitized_lines.pop(0)
    while sanitized_lines and not sanitized_lines[-1]:
        sanitized_lines.pop()

    return "\n".join(sanitized_lines)


def _escape_pdf_string(text: str) -> str:
    safe = text.encode("latin-1", errors="replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_lines(text: str) -> List[str]:
    wrapped: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            wrapped.append("")
            continue
        chunks = textwrap.wrap(
            line,
            width=_MAX_LINE_WIDTH,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(chunk.rstrip() for chunk in chunks if chunk.strip())
    return wrapped


def _paginate_lines(lines: List[str]) -> List[List[str]]:
    if not lines:
        return [[]]
    pages: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if len(current) >= _MAX_LINES_PER_PAGE:
            pages.append(current)
            current = []
        current.append(line)
    if current or not pages:
        pages.append(current)
    return pages


def _build_content_stream(lines: List[str]) -> bytes:
    commands: List[str] = [
        "BT",
        "/F1 11 Tf",
        "54 756 Td",
        "14 TL",
    ]
    for line in lines:
        if not line:
            commands.append("T*")
            continue
        commands.append(f"({_escape_pdf_string(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return ("\n".join(commands) + "\n").encode("latin-1", errors="replace")


def _build_pdf_bytes_from_lines(pages: List[List[str]]) -> bytes:
    page_count = max(1, len(pages))
    objects: List[bytes] = []
    kids: List[str] = []

    for page_index in range(page_count):
        page_obj_id = 3 + page_index * 2
        content_obj_id = page_obj_id + 1
        kids.append(f"{page_obj_id} 0 R")

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids_value = " ".join(kids)
    objects.append(
        f"<< /Type /Pages /Count {page_count} /Kids [{kids_value}] >>".encode("latin-1")
    )

    for page_index in range(page_count):
        page_obj_id = 3 + page_index * 2
        content_obj_id = page_obj_id + 1
        page_dictionary = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/Contents {content_obj_id} 0 R >>"
        ).encode("latin-1")
        stream_bytes = _build_content_stream(pages[page_index])
        stream_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"endstream"
        )
        objects.append(page_dictionary)
        objects.append(stream_obj)

    body = bytearray()
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body.extend(header)
    offsets: List[int] = [0]

    for object_id, payload in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{object_id} 0 obj\n".encode("latin-1"))
        body.extend(payload)
        body.extend(b"\nendobj\n")

    xref_offset = len(body)
    size = len(objects) + 1
    body.extend(f"xref\n0 {size}\n".encode("latin-1"))
    body.extend(b"0000000000 65535 f \n")
    for object_id in range(1, size):
        body.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    )
    body.extend(trailer.encode("latin-1"))
    return bytes(body)


def build_pdf_document_bytes_from_text(text: str) -> bytes:
    """Build deterministic PDF bytes from already-sanitized export text."""
    clean = _sanitize_plain_text(text)
    if not clean:
        clean = "ContentBlitz Export"
    wrapped = _wrap_lines(clean)
    pages = _paginate_lines(wrapped)
    return _build_pdf_bytes_from_lines(pages)


def build_pdf_export_document(state: Mapping[str, Any]) -> bytes:
    """Build deterministic PDF bytes from workflow state."""
    text = _build_pdf_export_text(state)
    return build_pdf_document_bytes_from_text(text)
