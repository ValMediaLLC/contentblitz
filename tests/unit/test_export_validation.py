from __future__ import annotations

from contentblitz.tools.exports.validation import (
    normalize_validation_result,
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

