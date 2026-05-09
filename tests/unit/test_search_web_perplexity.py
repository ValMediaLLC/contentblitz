from __future__ import annotations

import importlib

from contentblitz.tools.provider_types import SearchResult, SearchWebResult

search_web_module = importlib.import_module("contentblitz.tools.search_web")
perplexity_module = importlib.import_module("contentblitz.tools.perplexity")


def _result(provider: str, degraded: bool, results: list[SearchResult]) -> SearchWebResult:
    return SearchWebResult(
        provider=provider,
        query="q",
        results=results,
        degraded=degraded,
        error=None if not degraded else {"code": "x"},
    )


def test_serp_success_does_not_call_perplexity(monkeypatch) -> None:
    serp_success = _result(
        "serp",
        False,
        [
            SearchResult(
                title="SERP Result",
                url="https://serp.example/1",
                snippet="A usable SERP snippet.",
                source="serp",
                published_at=None,
                citation_available=True,
                credibility_score=0.8,
            )
        ],
    )
    calls = {"perplexity": 0}

    monkeypatch.setattr(search_web_module, "_search_serp", lambda query, max_results: serp_success)

    def _perplexity_should_not_run(query: str, max_results: int):
        calls["perplexity"] += 1
        raise AssertionError("Perplexity should not run when SERP is usable.")

    monkeypatch.setattr(search_web_module, "_search_perplexity_placeholder", _perplexity_should_not_run)

    result = search_web_module.search_web("ai workflows", provider="auto")
    assert result.provider == "serp"
    assert result.degraded is False
    assert calls["perplexity"] == 0


def test_serp_degraded_calls_perplexity(monkeypatch) -> None:
    serp_degraded = _result("serp", True, [])
    perplexity_success = _result(
        "perplexity",
        False,
        [
            SearchResult(
                title="Perplexity answer",
                url=None,
                snippet="Perplexity fallback snippet.",
                source="perplexity",
                published_at=None,
                citation_available=False,
                credibility_score=0.35,
            )
        ],
    )
    calls = {"perplexity": 0}

    monkeypatch.setattr(search_web_module, "_search_serp", lambda query, max_results: serp_degraded)

    def _perplexity(query: str, max_results: int):
        calls["perplexity"] += 1
        return perplexity_success

    monkeypatch.setattr(search_web_module, "_search_perplexity_placeholder", _perplexity)

    result = search_web_module.search_web("ai workflows", provider="auto")
    assert calls["perplexity"] == 1
    assert result.provider == "perplexity"
    assert result.degraded is False


def test_serp_exception_calls_perplexity(monkeypatch) -> None:
    calls = {"perplexity": 0}

    def _serp_raises(query: str, max_results: int):
        raise RuntimeError("serp exploded")

    def _perplexity(query: str, max_results: int):
        calls["perplexity"] += 1
        return _result(
            "perplexity",
            False,
            [
                SearchResult(
                    title="Fallback",
                    url=None,
                    snippet="fallback snippet",
                    source="perplexity",
                    published_at=None,
                    citation_available=False,
                    credibility_score=0.35,
                )
            ],
        )

    monkeypatch.setattr(search_web_module, "_search_serp", _serp_raises)
    monkeypatch.setattr(search_web_module, "_search_perplexity_placeholder", _perplexity)

    result = search_web_module.search_web("ai workflows", provider="auto")
    assert calls["perplexity"] == 1
    assert result.provider == "perplexity"
    assert result.degraded is False


def test_perplexity_success_returns_normalized_fallback_result(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "px-test-key")
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **kwargs: {
            "choices": [{"message": {"content": "Perplexity answer text"}}],
            "citations": ["https://source.example/a", "https://source.example/b"],
        },
    )

    result = perplexity_module.search_perplexity("ai workflows", max_results=2)
    assert result.provider == "perplexity"
    assert result.degraded is False
    assert len(result.results) == 2
    first = result.results[0]
    assert first.source == "perplexity"
    assert first.url == "https://source.example/a"
    assert first.citation_available is True
    assert first.published_at is None


def test_perplexity_missing_url_sets_citation_available_false(monkeypatch) -> None:
    monkeypatch.setenv("PERPLEXITY_API_KEY", "px-test-key")
    monkeypatch.setattr(
        perplexity_module,
        "_http_post_json",
        lambda **kwargs: {
            "choices": [{"message": {"content": "Perplexity no-url answer"}}],
            "citations": [],
        },
    )

    result = perplexity_module.search_perplexity("ai workflows", max_results=3)
    assert result.degraded is False
    assert len(result.results) == 1
    item = result.results[0]
    assert item.source == "perplexity"
    assert item.url is None
    assert item.citation_available is False
    assert item.published_at is None


def test_both_providers_failing_returns_degraded_result(monkeypatch) -> None:
    monkeypatch.setattr(
        search_web_module,
        "_search_serp",
        lambda query, max_results: _result("serp", True, []),
    )
    monkeypatch.setattr(
        search_web_module,
        "_search_perplexity_placeholder",
        lambda query, max_results: _result("perplexity", True, []),
    )

    result = search_web_module.search_web("ai workflows", provider="auto")
    assert result.degraded is True
    assert result.results == []
    assert result.error is not None
    assert result.error["code"] == "all_providers_failed"
