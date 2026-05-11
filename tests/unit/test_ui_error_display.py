from __future__ import annotations

from contentblitz.ui.error_display import (
    normalize_error_for_display,
    normalize_errors_for_display,
    redact_sensitive_text,
)


def test_api_key_like_values_are_redacted() -> None:
    raw = (
        "OPENAI_API_KEY=sk-test-secret-value "
        "SERP_API_KEY=serp_secret_12345 "
        "PERPLEXITY_API_KEY=pplx-very-secret-value"
    )
    redacted = redact_sensitive_text(raw)
    assert "sk-test-secret-value" not in redacted
    assert "serp_secret_12345" not in redacted
    assert "pplx-very-secret-value" not in redacted
    assert "[REDACTED]" in redacted


def test_stack_trace_message_is_normalized_to_safe_output() -> None:
    normalized = normalize_error_for_display(
        {
            "message": "Traceback (most recent call last):\n  File 'x.py', line 1\nValueError: bad",
            "recoverable": False,
        }
    )
    assert "Traceback" not in normalized["message"]
    assert "File 'x.py'" not in normalized["message"]
    assert normalized["message"]


def test_normalize_error_for_display_preserves_safe_structure() -> None:
    normalized = normalize_error_for_display(
        {
            "code": "provider_failure",
            "message": "Provider request failed cleanly.",
            "recoverable": True,
            "agent": "research_agent",
            "provider": "serp",
            "raw_payload": {"secret": "should-not-be-exposed"},
        }
    )
    assert normalized["code"] == "provider_failure"
    assert normalized["recoverable"] is True
    assert normalized["agent"] == "research_agent"
    assert normalized["provider"] == "serp"
    assert "raw_payload" not in normalized


def test_normalize_errors_for_display_handles_non_list_input() -> None:
    assert normalize_errors_for_display(None) == []
