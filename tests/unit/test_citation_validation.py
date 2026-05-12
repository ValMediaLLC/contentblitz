from __future__ import annotations

from contentblitz.quality.citations import (
    CITATION_VALIDATION_WARNING,
    validate_citation_sources,
)


def test_valid_citations_pass_without_warning() -> None:
    result = validate_citation_sources(
        [
            {
                "title": "AI Market Outlook 2026",
                "url": "https://example.com/ai-market-outlook",
                "snippet": "Market outlook cites sustained enterprise adoption growth.",
                "citation_available": True,
            }
        ],
        research_requested=True,
    )

    assert result["status"] == "passed"
    assert result["invalid_count"] == 0
    assert result["duplicate_count"] == 0
    assert result["unsafe_url_count"] == 0
    assert result["warning"] == ""
    assert result["valid_source_count"] == 1


def test_missing_title_or_snippet_is_detected() -> None:
    result = validate_citation_sources(
        [
            {"title": "", "url": "https://example.com/source-1", "snippet": "Useful snippet"},
            {"title": "Source 2", "url": "https://example.com/source-2", "snippet": ""},
        ],
        research_requested=True,
    )

    assert result["status"] == "degraded"
    assert result["missing_count"] >= 2
    assert result["invalid_count"] >= 2
    assert result["warning"] == CITATION_VALIDATION_WARNING


def test_duplicate_url_is_detected() -> None:
    result = validate_citation_sources(
        [
            {
                "title": "Source A",
                "url": "https://example.com/duplicate",
                "snippet": "A credible source.",
            },
            {
                "title": "Source B",
                "url": "https://example.com/duplicate",
                "snippet": "Duplicate URL entry.",
            },
        ],
        research_requested=True,
    )

    assert result["status"] == "degraded"
    assert result["duplicate_count"] >= 1
    assert result["valid_source_count"] == 1


def test_duplicate_title_and_url_is_detected() -> None:
    result = validate_citation_sources(
        [
            {
                "title": "Source A",
                "url": "https://example.com/same",
                "snippet": "A credible source.",
            },
            {
                "title": "Source A",
                "url": "https://example.com/same",
                "snippet": "Same title and URL.",
            },
        ],
        research_requested=True,
    )

    assert result["status"] == "degraded"
    assert result["duplicate_count"] >= 1
    assert result["valid_source_count"] == 1


def test_invalid_url_is_downgraded_safely() -> None:
    result = validate_citation_sources(
        [
            {
                "title": "Source A",
                "url": "javascript:alert(1)",
                "snippet": "Unsafe URL should be dropped.",
                "citation_available": True,
            }
        ],
        research_requested=True,
    )

    assert result["status"] == "degraded"
    assert result["unsafe_url_count"] == 1
    assert result["valid_source_count"] == 1
    assert result["sanitized_sources"][0]["url"] is None
    assert result["sanitized_sources"][0]["citation_available"] is False


def test_empty_sources_with_research_requested_surfaces_warning() -> None:
    result = validate_citation_sources([], research_requested=True)

    assert result["status"] == "degraded"
    assert result["warning"] == CITATION_VALIDATION_WARNING
    assert result["valid_source_count"] == 0


def test_raw_provider_payload_like_citation_text_is_rejected() -> None:
    result = validate_citation_sources(
        [
            {
                "title": "Source A",
                "url": "https://example.com/source-a",
                "snippet": "{'code': 'configuration_error', 'provider': 'openai'}",
            }
        ],
        research_requested=True,
    )

    assert result["status"] == "degraded"
    assert result["invalid_count"] >= 1
    assert result["valid_source_count"] == 0


def test_none_null_stacktrace_key_and_base64_leakage_is_rejected() -> None:
    result = validate_citation_sources(
        [
            {
                "title": "None",
                "url": "https://example.com/source-a",
                "snippet": "valid snippet",
            },
            {
                "title": "Source B",
                "url": "https://example.com/source-b",
                "snippet": "Traceback (most recent call last): ...",
            },
            {
                "title": "Source C",
                "url": "https://example.com/source-c",
                "snippet": "OPENAI_API_KEY should never appear",
            },
            {
                "title": "Source D",
                "url": "https://example.com/source-d",
                "snippet": "data:image/png;base64,AAAA",
            },
        ],
        research_requested=True,
    )

    assert result["status"] == "degraded"
    assert result["invalid_count"] >= 4
    assert result["valid_source_count"] == 0

