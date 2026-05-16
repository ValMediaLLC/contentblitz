from __future__ import annotations

from contentblitz.tools.exports.validation import (
    validate_docx_export,
    normalize_validation_result,
    validate_html_export,
    validate_markdown_export,
    validate_pdf_export,
)
from contentblitz.tools.exports.docx import build_docx_document_bytes_from_text
from contentblitz.quality.citations import CITATION_VALIDATION_WARNING
import io
import zipfile


def test_validate_markdown_export_accepts_safe_markdown() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`

## Blog Draft
Hello world.

## Sources
1. [Example](https://example.com)
"""
    result = validate_markdown_export(markdown, sources_exist=True)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_markdown_export_rejects_sensitive_payloads() -> None:
    markdown = """# ContentBlitz Export

Traceback (most recent call last):
OPENAI_API_KEY=sk-secret
data:image/png;base64,AAAA
"""
    result = validate_markdown_export(markdown, sources_exist=False)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "stack trace" in joined
    assert "environment variable" in joined
    assert "base64" in joined


def test_validate_markdown_export_rejects_raw_provider_payload_strings() -> None:
    markdown = """# ContentBlitz Export

## Image Outputs
- `failed` | `dall-e-3` | {'code': 'configuration_error', 'message': '[REDACTED] is not configured.', 'provider': 'openai', 'recoverable': False}
"""
    result = validate_markdown_export(markdown, sources_exist=False)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "raw provider/configuration payload" in joined


def test_validate_markdown_export_rejects_unsafe_urls() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`

## Sources
1. [Bad](javascript:alert(1))
"""
    result = validate_markdown_export(markdown, sources_exist=True)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "unsafe or invalid url" in joined


def test_validate_markdown_export_rejects_invalid_source_entries() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`

## Sources
1.
"""
    result = validate_markdown_export(markdown, sources_exist=True)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "readable citations" in joined


def test_validate_markdown_export_accepts_bracketed_source_entries() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`

## Sources
[1] Export source (https://example.com/export-source)
"""
    result = validate_markdown_export(markdown, sources_exist=True)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_markdown_export_requires_sources_section_when_sources_exist() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`
"""
    result = validate_markdown_export(markdown, sources_exist=True)
    assert result["valid"] is False
    assert any(
        "sources section is required" in error.lower() for error in result["errors"]
    )


def test_validate_markdown_export_warns_when_structured_citations_are_invalid() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `partial_success`

## Sources
1. [Example](https://example.com)
"""
    result = validate_markdown_export(
        markdown,
        sources_exist=True,
        sources=[
            {
                "title": "Source A",
                "url": "javascript:alert(1)",
                "snippet": "Unsafe URL should be downgraded.",
            }
        ],
    )

    assert result["valid"] is True
    assert CITATION_VALIDATION_WARNING in result["warnings"]


def test_validate_markdown_export_has_no_citation_warning_for_valid_structured_sources() -> (
    None
):
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`

## Sources
1. [Example](https://example.com)
"""
    result = validate_markdown_export(
        markdown,
        sources_exist=True,
        sources=[
            {
                "title": "Example",
                "url": "https://example.com",
                "snippet": "Safe source snippet.",
            }
        ],
    )

    assert result["valid"] is True
    assert CITATION_VALIDATION_WARNING not in result["warnings"]


def test_normalize_validation_result_strips_empty_fields() -> None:
    normalized = normalize_validation_result(
        {
            "valid": "yes",
            "warnings": ["", None, "warning"],
            "errors": ["", "error"],
        }
    )
    assert normalized == {
        "valid": True,
        "warnings": ["warning"],
        "errors": ["error"],
    }


def test_validate_html_export_accepts_safe_html() -> None:
    html = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>ContentBlitz Export</title></head>
<body>
  <h1>ContentBlitz Export</h1>
  <h2>Workflow Summary</h2>
  <h2>Sources</h2>
  <ul><li><a href="https://example.com">Example</a></li></ul>
</body>
</html>
"""
    result = validate_html_export(html, sources_exist=True)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_html_export_rejects_unsafe_html_payloads() -> None:
    html = """<!doctype html>
<html>
<body onload="alert(1)">
<script>alert(1)</script>
<a href="javascript:alert(1)">x</a>
OPENAI_API_KEY=sk-secret
<iframe src="https://evil.example"></iframe>
</body>
</html>
"""
    result = validate_html_export(html, sources_exist=False)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "script tags" in joined
    assert "inline javascript handlers" in joined
    assert "javascript: urls" in joined
    assert "environment variable" in joined
    assert "unsafe embed tags" in joined


def test_validate_html_export_rejects_invalid_urls() -> None:
    html = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>ContentBlitz Export</title></head>
<body>
  <h1>ContentBlitz Export</h1>
  <h2>Workflow Summary</h2>
  <h2>Sources</h2>
  <ul><li><a href="file:///etc/passwd">bad</a></li></ul>
</body>
</html>
"""
    result = validate_html_export(html, sources_exist=True)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "unsafe or invalid url" in joined


def test_validate_markdown_export_rejects_unsupported_schemes() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`

## Sources
1. [FTP](ftp://example.com/source)
2. [Mail](mailto:test@example.com)
3. [VBS](vbscript:alert(1))
"""
    result = validate_markdown_export(markdown, sources_exist=True)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "unsafe or invalid url" in joined


def test_validate_pdf_export_accepts_safe_pdf() -> None:
    payload = (
        b"%PDF-1.4\n"
        b"ContentBlitz Export\nWorkflow Summary\nSources\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Count 0 /Kids [] >>\nendobj\n"
        b"xref\n0 3\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n"
        b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n100\n%%EOF\n"
    )
    result = validate_pdf_export(payload, sources_exist=False)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_pdf_export_rejects_sensitive_payloads() -> None:
    payload = (
        b"%PDF-1.4\n"
        b"Traceback (most recent call last):\n"
        b"OPENAI_API_KEY=sk-secret\n"
        b"data:image/png;base64,AAAA\n"
        b"<script>alert(1)</script>\n"
        b'<a href="javascript:alert(1)">x</a>\n'
        b"xref\ntrailer\n%%EOF\n"
    )
    result = validate_pdf_export(payload, sources_exist=False)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "stack trace" in joined
    assert "environment variable" in joined
    assert "base64" in joined
    assert "script tags" in joined
    assert "javascript: urls" in joined


def test_validate_docx_export_accepts_safe_docx() -> None:
    payload = build_docx_document_bytes_from_text(
        "# ContentBlitz Export\n\n## Workflow Summary\n- Workflow Status: success\n\n## Sources\n1. Example"
    )
    result = validate_docx_export(payload, sources_exist=False)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_docx_export_rejects_sensitive_payloads() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body><w:p><w:r><w:t>Traceback (most recent call last): OPENAI_API_KEY=sk-secret data:image/png;base64,AAAA</w:t></w:r></w:p></w:body>"
                "</w:document>"
            ),
        )

    result = validate_docx_export(buffer.getvalue(), sources_exist=False)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "stack trace" in joined
    assert "environment variable" in joined
    assert "base64" in joined
