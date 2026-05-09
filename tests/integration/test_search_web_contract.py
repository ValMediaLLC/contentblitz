from __future__ import annotations

import importlib
from dataclasses import is_dataclass

search_web_module = importlib.import_module("contentblitz.tools.search_web")
legacy_search_web_module = importlib.import_module("contentblitz.tools.web_search")


def test_search_web_contract_shape_is_stable(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-contract-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {
            "organic_results": [
                {
                    "title": "Contract Result",
                    "link": "https://contract.example/result",
                    "snippet": "Contract test snippet",
                    "source": "ContractSource",
                    "date": "2026-05-01",
                }
            ]
        },
    )

    result = search_web_module.search_web(
        "contract query",
        max_results=3,
        provider="serp",
    )

    assert is_dataclass(result)
    assert result.provider == "serp"
    assert result.query == "contract query"
    assert isinstance(result.results, list)
    assert result.degraded is False
    assert result.error is None
    assert len(result.results) == 1

    item = result.results[0]
    assert item.title == "Contract Result"
    assert item.url == "https://contract.example/result"
    assert item.snippet == "Contract test snippet"
    assert item.source == "ContractSource"
    assert item.published_at == "2026-05-01"
    assert item.citation_available is True
    assert isinstance(item.credibility_score, float)


def test_legacy_web_search_adapter_shape_remains_stable(monkeypatch) -> None:
    monkeypatch.setenv("SERP_API_KEY", "serp-contract-key")
    monkeypatch.setattr(
        search_web_module,
        "_http_get_json",
        lambda url: {
            "organic_results": [
                {
                    "title": "Legacy Result",
                    "link": "https://legacy.example/result",
                    "snippet": "Legacy snippet",
                    "source": "LegacySource",
                    "date": "May 2026",
                }
            ]
        },
    )

    legacy = legacy_search_web_module.search_web(
        query="legacy contract query",
        depth="standard",
    )

    assert legacy["query"] == "legacy contract query"
    assert legacy["depth"] == "standard"
    assert legacy["provider_primary"] == "serp_api"
    assert legacy["provider_fallback"] == "perplexity"
    assert legacy["provider_used"] == "serp"
    assert legacy["degraded"] is False
    assert legacy["error"] is None
    assert isinstance(legacy["results"], list)
    assert len(legacy["results"]) == 1

    item = legacy["results"][0]
    assert item["title"] == "Legacy Result"
    assert item["url"] == "https://legacy.example/result"
    assert item["snippet"] == "Legacy snippet"
    assert item["source"] == "LegacySource"
    assert item["date"] == "May 2026"
    assert item["citation_available"] is True
    assert isinstance(item["credibility_score"], float)
