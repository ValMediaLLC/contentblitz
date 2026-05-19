from __future__ import annotations

from copy import deepcopy

from contentblitz.core.redaction import (
    REDACTED_BASE64_PAYLOAD,
    REDACTED_RAW_PAYLOAD,
    REDACTED_STACK_TRACE,
    REDACTED_TOKEN,
    TRUNCATED_ITEMS_TOKEN,
    TRUNCATED_SUFFIX,
    redact_sensitive_text,
    sanitize_trace_value,
)


def test_redacts_api_key_like_strings() -> None:
    text = (
        "OPENAI_API_KEY=sk-super-secret "
        "LANGSMITH_API_KEY=lsv2_very_secret "
        "SERP_API_KEY=serp_hidden "
        "PERPLEXITY_API_KEY=pplx-hidden"
    )
    redacted = redact_sensitive_text(text)

    assert REDACTED_TOKEN in redacted
    assert "sk-super-secret" not in redacted
    assert "lsv2_very_secret" not in redacted
    assert "serp_hidden" not in redacted
    assert "pplx-hidden" not in redacted


def test_redacts_bearer_tokens() -> None:
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def"
    redacted = redact_sensitive_text(text)

    assert "Bearer [REDACTED]" in redacted
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def" not in redacted


def test_redacts_stack_trace_like_content() -> None:
    text = (
        "Traceback (most recent call last):\n"
        '  File "service.py", line 42, in run\n'
        "ValueError: bad"
    )
    redacted = redact_sensitive_text(text)

    assert redacted == REDACTED_STACK_TRACE


def test_normal_markdown_preview_is_not_misclassified_as_stack_trace() -> None:
    text = (
        "## Blog Summary\n"
        "This line describes the workflow outcome and references inline sources.\n"
        "Final line includes a call to action."
    )
    redacted = redact_sensitive_text(text, max_length=240)

    assert redacted != REDACTED_STACK_TRACE
    assert "## Blog Summary" in redacted
    assert "line describes" in redacted


def test_redacts_base64_image_payloads() -> None:
    text = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
    redacted = redact_sensitive_text(text)

    assert REDACTED_BASE64_PAYLOAD in redacted
    assert "iVBORw0KGgoAAAANSUhEUgAAAAUA" not in redacted


def test_truncates_oversized_metadata_values() -> None:
    text = "x" * 2000
    redacted = redact_sensitive_text(text, max_length=120)

    assert len(redacted) <= 120 + len(TRUNCATED_SUFFIX)
    assert redacted.endswith(TRUNCATED_SUFFIX)


def test_safe_metadata_values_are_preserved() -> None:
    value = {
        "node_name": "query_handler_node",
        "node_status": "completed",
        "workflow_status": "success",
        "requested_outputs": ["blog", "linkedin"],
        "source_count": 4,
        "provider_degraded": False,
    }

    sanitized = sanitize_trace_value(value)

    assert sanitized["node_name"] == "query_handler_node"
    assert sanitized["node_status"] == "completed"
    assert sanitized["workflow_status"] == "success"
    assert sanitized["requested_outputs"] == ["blog", "linkedin"]
    assert sanitized["source_count"] == 4
    assert sanitized["provider_degraded"] is False


def test_redaction_helper_does_not_mutate_input_state() -> None:
    value = {
        "token": "OPENAI_API_KEY=sk-test",
        "nested": {"auth": "Bearer secret-token-value"},
        "items": ["safe", "data:image/png;base64,AAAA"],
    }
    original = deepcopy(value)

    _ = sanitize_trace_value(value)

    assert value == original


def test_sanitize_trace_value_redacts_raw_payload_like_mapping_keys() -> None:
    value = {
        "raw_request_payload": {"prompt": "do not leak"},
        "safe_key": "safe",
    }
    sanitized = sanitize_trace_value(value)

    assert sanitized["raw_request_payload"] == REDACTED_RAW_PAYLOAD
    assert sanitized["safe_key"] == "safe"


def test_sanitize_trace_value_truncates_oversized_lists() -> None:
    items = list(range(60))
    sanitized = sanitize_trace_value(items)

    assert TRUNCATED_ITEMS_TOKEN in sanitized
