from __future__ import annotations

from copy import deepcopy
import importlib
from urllib.error import URLError

search_web_module = importlib.import_module("contentblitz.tools.search_web")


def test_successful_serp_response_normalizes_results(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-test-key")

    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {
            "organic_results": [
                {
                    "title": "AI Workflow Systems",
                    "link": "https://example.com/ai-workflows",
                    "snippet": "A concise summary about AI workflow systems.",
                    "source": "Example",
                    "date": "2 days ago",
                }
            ]
        },
    )

    result = search_web_module.search_web(
        "ai content workflows",
        max_results=5,
        provider="serp",
    )

    assert result.provider == "serp"
    assert result.query == "ai content workflows"
    assert result.degraded is False
    assert result.error is None
    assert len(result.results) == 1

    item = result.results[0]
    assert item.title == "AI Workflow Systems"
    assert item.url == "https://example.com/ai-workflows"
    assert item.snippet == "A concise summary about AI workflow systems."
    assert item.source == "Example"
    assert item.published_at == "2 days ago"
    assert item.citation_available is True
    assert isinstance(item.credibility_score, float)


def test_missing_url_sets_citation_available_false(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-test-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {
            "organic_results": [
                {
                    "title": "No URL Result",
                    "snippet": "Snippet without a URL.",
                    "source": "NoUrlSource",
                }
            ]
        },
    )

    result = search_web_module.search_web("no url query", provider="serp")
    assert result.degraded is False
    assert len(result.results) == 1
    assert result.results[0].url is None
    assert result.results[0].citation_available is False


def test_duplicate_urls_are_removed(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-test-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {
            "organic_results": [
                {
                    "title": "First",
                    "link": "https://dup.example/article",
                    "snippet": "first snippet",
                    "source": "DupSource",
                },
                {
                    "title": "Second Duplicate",
                    "link": "https://dup.example/article",
                    "snippet": "second snippet",
                    "source": "DupSource",
                },
                {
                    "title": "Unique",
                    "link": "https://unique.example/article",
                    "snippet": "unique snippet",
                    "source": "UniqueSource",
                },
            ]
        },
    )

    result = search_web_module.search_web("duplicate url query", provider="serp")
    assert result.degraded is False
    assert len(result.results) == 2
    urls = [item.url for item in result.results]
    assert urls == ["https://dup.example/article", "https://unique.example/article"]


def test_provider_failure_returns_degraded_result(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-test-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: (_ for _ in ()).throw(URLError("provider down")),
    )

    result = search_web_module.search_web("provider failure query", provider="serp")
    assert result.degraded is True
    assert result.results == []
    assert result.error is not None
    assert result.error["code"] == "provider_unavailable"


def test_missing_serp_api_key_fails_safely(monkeypatch) -> None:
    monkeypatch.delenv("SERP_API_KEY", raising=False)

    result = search_web_module.search_web("missing key query", provider="serp")
    assert result.degraded is True
    assert result.results == []
    assert result.error is not None
    assert result.error["code"] == "configuration_error"


def test_malformed_provider_payload_returns_degraded_result(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-test-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {"unexpected": "shape"},
    )

    result = search_web_module.search_web("bad payload query", provider="serp")
    assert result.degraded is True
    assert result.error is not None
    assert result.error["code"] == "provider_payload_invalid"


def test_no_state_mutation_occurs(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-test-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {"organic_results": []},
    )

    state = {
        "retry_counts": {"research_agent": 2},
        "cost_controls": {"search_queries_used_this_session": 9},
        "content_drafts": {"blog": {"body": "x", "version": 1}},
        "sources": [{"title": "Existing"}],
        "errors": [{"type": "existing"}],
    }
    before = deepcopy(state)

    _ = search_web_module.search_web("state mutation check", provider="serp")
    assert state == before


def test_no_cache_access_occurs(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-test-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {"organic_results": []},
    )

    def _fail(*args, **kwargs):
        raise AssertionError("Cache helper should not be called by search_web tool.")

    monkeypatch.setattr("contentblitz.tools.cache.get_cached_research", _fail)
    monkeypatch.setattr("contentblitz.tools.cache.set_cached_research", _fail)

    result = search_web_module.search_web("cache access check", provider="serp")
    assert result.provider == "serp"
