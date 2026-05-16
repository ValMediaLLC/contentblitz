"""PDF export renderer and sanitizer for ContentBlitz."""

from __future__ import annotations

import binascii
import re
import struct
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

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
_PAGE_WIDTH = 612.0
_PAGE_HEIGHT = 792.0
_PAGE_MARGIN_X = 54.0
_PAGE_MARGIN_TOP = 54.0
_PAGE_MARGIN_BOTTOM = 72.0
_IMAGE_MAX_WIDTH = _PAGE_WIDTH - (_PAGE_MARGIN_X * 2.0)
_IMAGE_MAX_HEIGHT = _PAGE_HEIGHT - _PAGE_MARGIN_TOP - _PAGE_MARGIN_BOTTOM - 32.0
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOI = b"\xff\xd8"
_GENERIC_IMAGE_EMBED_WARNING = (
    "A renderable local image could not be embedded; text export was preserved."
)


@dataclass(frozen=True)
class _EmbeddedImage:
    width: int
    height: int
    color_space: str
    bits_per_component: int
    filter_name: str
    stream_data: bytes
    decode_parms: str = ""


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


def _is_unsafe_image_ref(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("data:image/") or "base64" in lowered


def _resolve_local_image_path(value: Any) -> Path | None:
    local_path = _safe_text(value)
    if not local_path or _is_unsafe_image_ref(local_path):
        return None
    try:
        candidate = Path(local_path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (Path.cwd() / candidate).resolve()
        )
    except Exception:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


def _read_png_chunks(png_bytes: bytes) -> List[Tuple[bytes, bytes]]:
    if not png_bytes.startswith(_PNG_SIGNATURE):
        raise ValueError("not_png")
    cursor = len(_PNG_SIGNATURE)
    chunks: List[Tuple[bytes, bytes]] = []
    total = len(png_bytes)
    while cursor + 12 <= total:
        length = struct.unpack(">I", png_bytes[cursor : cursor + 4])[0]
        chunk_type = png_bytes[cursor + 4 : cursor + 8]
        data_start = cursor + 8
        data_end = data_start + length
        crc_start = data_end
        crc_end = crc_start + 4
        if crc_end > total:
            raise ValueError("png_truncated")
        chunk_data = png_bytes[data_start:data_end]
        expected_crc = struct.unpack(">I", png_bytes[crc_start:crc_end])[0]
        actual_crc = binascii.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ValueError("png_crc")
        chunks.append((chunk_type, chunk_data))
        cursor = crc_end
        if chunk_type == b"IEND":
            break
    return chunks


def _parse_png_embedded_image(png_bytes: bytes) -> _EmbeddedImage:
    chunks = _read_png_chunks(png_bytes)
    width = 0
    height = 0
    bit_depth = 0
    color_type = -1
    idat_parts: List[bytes] = []

    for chunk_type, chunk_data in chunks:
        if chunk_type == b"IHDR":
            if len(chunk_data) != 13:
                raise ValueError("png_ihdr")
            width = struct.unpack(">I", chunk_data[0:4])[0]
            height = struct.unpack(">I", chunk_data[4:8])[0]
            bit_depth = int(chunk_data[8])
            color_type = int(chunk_data[9])
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)

    if width <= 0 or height <= 0 or bit_depth <= 0 or color_type < 0:
        raise ValueError("png_header_missing")
    if not idat_parts:
        raise ValueError("png_idat_missing")
    if bit_depth != 8:
        raise ValueError("png_unsupported_bits")

    if color_type == 0:
        color_space = "/DeviceGray"
        colors = 1
    elif color_type == 2:
        color_space = "/DeviceRGB"
        colors = 3
    else:
        raise ValueError("png_unsupported_color_type")

    idat_stream = b"".join(idat_parts)
    decode_parms = (
        f"/DecodeParms << /Predictor 15 /Colors {colors} "
        f"/BitsPerComponent {bit_depth} /Columns {width} >>"
    )
    return _EmbeddedImage(
        width=width,
        height=height,
        color_space=color_space,
        bits_per_component=bit_depth,
        filter_name="/FlateDecode",
        stream_data=idat_stream,
        decode_parms=decode_parms,
    )


def _parse_jpeg_dimensions(jpeg_bytes: bytes) -> Tuple[int, int, str]:
    if len(jpeg_bytes) < 4 or not jpeg_bytes.startswith(_JPEG_SOI):
        raise ValueError("not_jpeg")
    index = 2
    length = len(jpeg_bytes)
    while index + 4 <= length:
        if jpeg_bytes[index] != 0xFF:
            index += 1
            continue
        marker = jpeg_bytes[index + 1]
        index += 2
        while marker == 0xFF and index < length:
            marker = jpeg_bytes[index]
            index += 1
        if marker in {0xD8, 0xD9}:
            continue
        if marker in {0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > length:
            break
        segment_length = struct.unpack(">H", jpeg_bytes[index : index + 2])[0]
        if segment_length < 2 or index + segment_length > length:
            raise ValueError("jpeg_segment")
        segment_start = index + 2
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length < 8:
                raise ValueError("jpeg_sof")
            precision = jpeg_bytes[segment_start]
            height = struct.unpack(
                ">H", jpeg_bytes[segment_start + 1 : segment_start + 3]
            )[0]
            width = struct.unpack(
                ">H", jpeg_bytes[segment_start + 3 : segment_start + 5]
            )[0]
            components = jpeg_bytes[segment_start + 5]
            if precision != 8 or width <= 0 or height <= 0:
                raise ValueError("jpeg_unsupported")
            color_space = "/DeviceGray" if components == 1 else "/DeviceRGB"
            return width, height, color_space
        index += segment_length
    raise ValueError("jpeg_sof_missing")


def _parse_jpeg_embedded_image(jpeg_bytes: bytes) -> _EmbeddedImage:
    width, height, color_space = _parse_jpeg_dimensions(jpeg_bytes)
    return _EmbeddedImage(
        width=width,
        height=height,
        color_space=color_space,
        bits_per_component=8,
        filter_name="/DCTDecode",
        stream_data=jpeg_bytes,
    )


def _parse_embedded_image_bytes(image_path: Path) -> _EmbeddedImage:
    payload = image_path.read_bytes()
    if not payload:
        raise ValueError("image_empty")
    if payload.startswith(_PNG_SIGNATURE):
        return _parse_png_embedded_image(payload)
    if payload.startswith(_JPEG_SOI):
        return _parse_jpeg_embedded_image(payload)
    raise ValueError("image_format_unsupported")


def _is_renderable_image_output(item: Mapping[str, Any]) -> bool:
    explicit = item.get("renderable")
    if isinstance(explicit, bool):
        return explicit
    return bool(_safe_text(item.get("local_path")) or _safe_text(item.get("url")))


def _collect_embedded_images_and_warnings(
    state: Mapping[str, Any],
) -> Tuple[List[_EmbeddedImage], List[str]]:
    embedded: List[_EmbeddedImage] = []
    warnings: List[str] = []
    image_outputs = state.get("image_outputs", [])
    if not isinstance(image_outputs, list):
        image_outputs = []
    for raw in image_outputs:
        if not isinstance(raw, Mapping):
            continue
        if not _is_renderable_image_output(raw):
            continue
        local_path = _safe_text(raw.get("local_path"))
        url = _safe_text(raw.get("url"))
        if local_path:
            resolved = _resolve_local_image_path(local_path)
            if resolved is None:
                warnings.append(_GENERIC_IMAGE_EMBED_WARNING)
                continue
            try:
                embedded.append(_parse_embedded_image_bytes(resolved))
            except Exception:
                warnings.append(_GENERIC_IMAGE_EMBED_WARNING)
            continue
        if url and not _is_unsafe_image_ref(url):
            # URL references remain text-only in PDF export by design.
            continue
    return embedded, list(dict.fromkeys(warnings))


def _append_warning_lines(text: str, warnings: Sequence[str]) -> str:
    if not warnings:
        return text
    lines = [line for line in text.splitlines()]
    lines.extend(
        [
            "",
            "Image Export Warnings:",
            *[
                f"- {_sanitize_plain_text(item)}"
                for item in warnings
                if _sanitize_plain_text(item)
            ],
        ]
    )
    return "\n".join(lines).strip()


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


def _scale_image_dimensions(width: int, height: int) -> Tuple[float, float]:
    if width <= 0 or height <= 0:
        return (_IMAGE_MAX_WIDTH, _IMAGE_MAX_HEIGHT)
    ratio = min(_IMAGE_MAX_WIDTH / float(width), _IMAGE_MAX_HEIGHT / float(height), 1.0)
    return (float(width) * ratio, float(height) * ratio)


def _build_image_content_stream(
    image_name: str,
    image_width: int,
    image_height: int,
    *,
    caption: str = "Embedded Image",
) -> bytes:
    draw_width, draw_height = _scale_image_dimensions(image_width, image_height)
    x = (_PAGE_WIDTH - draw_width) / 2.0
    y = _PAGE_HEIGHT - _PAGE_MARGIN_TOP - draw_height
    caption_safe = _escape_pdf_string(_sanitize_plain_text(caption) or "Embedded Image")
    commands = [
        "BT",
        "/F1 11 Tf",
        f"{_PAGE_MARGIN_X:.2f} {_PAGE_HEIGHT - _PAGE_MARGIN_TOP + 8.0:.2f} Td",
        f"({caption_safe}) Tj",
        "ET",
        "q",
        f"{draw_width:.2f} 0 0 {draw_height:.2f} {x:.2f} {y:.2f} cm",
        f"/{image_name} Do",
        "Q",
    ]
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
            "/Resources << "
            "/Font << /F1 << /Type /Font /Subtype /Type1 "
            "/BaseFont /Helvetica >> >> "
            ">> "
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


def _build_pdf_bytes_from_lines_and_images(
    text_pages: List[List[str]],
    images: Sequence[_EmbeddedImage],
) -> bytes:
    normalized_text_pages = text_pages if text_pages else [[]]
    text_page_count = len(normalized_text_pages)
    image_count = len(images)
    page_count = text_page_count + image_count

    objects: List[bytes] = []
    kids: List[str] = []

    page_object_ids: List[int] = []
    page_content_ids: List[int] = []
    next_id = 3
    for _ in range(page_count):
        page_object_ids.append(next_id)
        page_content_ids.append(next_id + 1)
        next_id += 2
    image_object_ids = [next_id + idx for idx in range(image_count)]

    for page_obj_id in page_object_ids:
        kids.append(f"{page_obj_id} 0 R")

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids_value = " ".join(kids)
    objects.append(
        f"<< /Type /Pages /Count {page_count} /Kids [{kids_value}] >>".encode("latin-1")
    )

    for page_index in range(text_page_count):
        page_obj_id = page_object_ids[page_index]
        content_obj_id = page_content_ids[page_index]
        page_dictionary = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << "
            "/Font << /F1 << /Type /Font /Subtype /Type1 "
            "/BaseFont /Helvetica >> >> "
            ">> "
            f"/Contents {content_obj_id} 0 R >>"
        ).encode("latin-1")
        stream_bytes = _build_content_stream(normalized_text_pages[page_index])
        stream_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"endstream"
        )
        objects.append(page_dictionary)
        objects.append(stream_obj)

    for image_index, image in enumerate(images):
        page_index = text_page_count + image_index
        page_obj_id = page_object_ids[page_index]
        content_obj_id = page_content_ids[page_index]
        image_obj_id = image_object_ids[image_index]
        image_name = f"Im{image_index + 1}"
        page_dictionary = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << "
            "/Font << /F1 << /Type /Font /Subtype /Type1 "
            "/BaseFont /Helvetica >> >> "
            f"/XObject << /{image_name} {image_obj_id} 0 R >> >> "
            f"/Contents {content_obj_id} 0 R >>"
        ).encode("latin-1")
        stream_bytes = _build_image_content_stream(
            image_name=image_name,
            image_width=image.width,
            image_height=image.height,
            caption=f"Embedded image {image_index + 1}",
        )
        stream_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"endstream"
        )
        objects.append(page_dictionary)
        objects.append(stream_obj)

    for image in images:
        decode_part = f" {image.decode_parms}" if image.decode_parms else ""
        image_dictionary = (
            f"<< /Type /XObject /Subtype /Image /Width {image.width} "
            f"/Height {image.height} /ColorSpace {image.color_space} "
            f"/BitsPerComponent {image.bits_per_component} "
            f"/Filter {image.filter_name}{decode_part} "
            f"/Length {len(image.stream_data)} >>\n"
        ).encode("latin-1")
        stream_obj = image_dictionary + b"stream\n" + image.stream_data + b"\nendstream"
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
    images, warnings = _collect_embedded_images_and_warnings(state)
    clean_text = _append_warning_lines(text, warnings)
    if not clean_text:
        clean_text = "ContentBlitz Export"
    wrapped = _wrap_lines(clean_text)
    pages = _paginate_lines(wrapped)
    return _build_pdf_bytes_from_lines_and_images(pages, images)
