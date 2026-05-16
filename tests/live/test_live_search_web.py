from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from contentblitz.tools.search_web import search_web


pytestmark = pytest.mark.skipif(
    os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") != "1",
    reason="Live provider tests are disabled by default.",
)


def _print_result(label: str, result) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print("Provider:", result.provider)
    print("Degraded:", result.degraded)
    print("Result count:", len(result.results))
    print("Error code:", result.error.get("code") if result.error else None)


def _assert_result_shape(result, *, max_results: int) -> None:
    assert isinstance(result.results, list)
    assert len(result.results) <= max_results
    for item in result.results:
        assert item.title
        assert isinstance(item.citation_available, bool)
        if item.citation_available:
            assert isinstance(item.url, str) and item.url.startswith(
                ("http://", "https://")
            )


def test_live_search_web_serp_provider():
    if not os.getenv("SERP_API_KEY"):
        pytest.skip("SERP_API_KEY is not set.")

    result = search_web(
        query="latest AI content workflow trends",
        max_results=3,
        provider="serp",
    )
    _print_result("SEARCH RESULT (SERP)", result)
    _assert_result_shape(result, max_results=3)

    assert result.provider == "serp"
    assert result.degraded is False
    assert len(result.results) > 0


def test_live_search_web_auto_provider():
    has_serp = bool(os.getenv("SERP_API_KEY"))
    has_perplexity = bool(os.getenv("PERPLEXITY_API_KEY"))
    if not has_serp and not has_perplexity:
        pytest.skip(
            "SERP_API_KEY or PERPLEXITY_API_KEY is required for provider='auto'."
        )

    result = search_web(
        query="best practices for AI content quality workflows",
        max_results=3,
        provider="auto",
    )
    _print_result("SEARCH RESULT (AUTO)", result)
    _assert_result_shape(result, max_results=3)

    assert result.provider in {"serp", "perplexity", "auto"}
    assert result.degraded is False
    assert len(result.results) > 0
