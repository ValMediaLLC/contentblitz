from __future__ import annotations

from contentblitz.tools.exports.validation import (
    normalize_validation_result,
    validate_html_export,
    validate_markdown_export,
)


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


def test_validate_markdown_export_requires_sources_section_when_sources_exist() -> None:
    markdown = """# ContentBlitz Export

## Workflow Summary
- Workflow Status: `success`
"""
    result = validate_markdown_export(markdown, sources_exist=True)
    assert result["valid"] is False
    assert any("sources section is required" in error.lower() for error in result["errors"])


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
