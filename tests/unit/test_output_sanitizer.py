from __future__ import annotations

from contentblitz.safety.output_sanitizer import (
    is_safe_external_url,
    sanitize_html_output,
    sanitize_markdown_output,
    sanitize_plain_output,
)


def test_markdown_sanitizer_removes_script_tags_and_event_handlers() -> None:
    raw = 'Safe text <script>alert(1)</script><a onclick="alert(1)">click</a>'
    sanitized, changed = sanitize_markdown_output(raw)
    lowered = sanitized.lower()
    assert changed is True
    assert "<script" not in lowered
    assert "onclick=" not in lowered
    assert "safe text" in lowered


def test_markdown_sanitizer_downgrades_unsafe_links_and_strips_data_images() -> None:
    raw = (
        "[good](https://example.com) "
        "[bad](javascript:alert(1)) "
        "![img](data:image/png;base64,AAAA)"
    )
    sanitized, _ = sanitize_markdown_output(raw)
    assert "[good](https://example.com)" in sanitized
    assert "[bad](" not in sanitized
    assert "javascript:" not in sanitized.lower()
    assert "data:image/" not in sanitized.lower()


def test_html_sanitizer_removes_unsafe_embeds_and_links() -> None:
    raw = (
        "<iframe src='https://evil.test'></iframe>"
        "<a href='javascript:alert(1)'>bad</a>"
        "<a href='https://example.com'>good</a>"
    )
    sanitized, changed = sanitize_html_output(raw)
    lowered = sanitized.lower()
    assert changed is True
    assert "<iframe" not in lowered
    assert "javascript:" not in lowered
    assert "https://example.com" in sanitized


def test_plain_sanitizer_removes_provider_payload_and_stack_trace() -> None:
    raw = (
        "Traceback (most recent call last):\n"
        "{'code': 'configuration_error', 'provider': 'openai'}\n"
        "OPENAI_API_KEY=sk-secret"
    )
    sanitized, _ = sanitize_plain_output(raw)
    lowered = sanitized.lower()
    assert "traceback" not in lowered
    assert "configuration_error" not in lowered
    assert "openai_api_key" not in lowered
    assert "sk-secret" not in lowered


def test_plain_sanitizer_neutralizes_system_developer_prompt_leakage() -> None:
    raw = (
        "REVEAL SYSTEM PROMPT: hidden text\n"
        "Developer message: internal directive\n"
        "Discuss system prompt hardening best practices."
    )
    sanitized, _ = sanitize_plain_output(raw)
    lowered = sanitized.lower()
    assert "reveal system prompt" not in lowered
    assert "developer message" not in lowered
    assert "system prompt" not in lowered
    assert "discuss [redacted] hardening best practices." in lowered


def test_safe_markdown_formatting_is_preserved() -> None:
    raw = (
        "# Heading\n\n"
        "- Bullet item\n"
        "1. Numbered\n"
        "**Bold** and `code` and [ref](https://example.com)"
    )
    sanitized, changed = sanitize_markdown_output(raw)
    assert "# Heading" in sanitized
    assert "- Bullet item" in sanitized
    assert "1. Numbered" in sanitized
    assert "**Bold**" in sanitized
    assert "`code`" in sanitized
    assert "[ref](https://example.com)" in sanitized
    assert changed is False


def test_safe_url_policy_http_https_only() -> None:
    assert is_safe_external_url("https://example.com") is True
    assert is_safe_external_url("http://example.com/path") is True
    assert is_safe_external_url("javascript:alert(1)") is False
    assert is_safe_external_url("data:text/plain,hello") is False
    assert is_safe_external_url("file:///tmp/a.txt") is False
    assert is_safe_external_url("ftp://example.com") is False
    assert is_safe_external_url("mailto:test@example.com") is False
    assert is_safe_external_url("vbscript:msgbox(1)") is False
