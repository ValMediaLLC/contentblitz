from contentblitz.agents import research_agent as research_agent_module


def test_stopwords_are_removed_from_keywords() -> None:
    query = "what is the best strategy for ai and seo in linkedin marketing"
    keywords = research_agent_module._build_keywords(query=query, sources=[])  # noqa: SLF001

    lowered = {keyword.lower() for keyword in keywords}
    assert "what" not in lowered
    assert "is" not in lowered
    assert "the" not in lowered
    assert "for" not in lowered
    assert "and" not in lowered
    assert "in" not in lowered


def test_meaningful_terms_are_preserved() -> None:
    keywords = research_agent_module._build_keywords(  # noqa: SLF001
        query="AI SEO LinkedIn marketing",
        sources=[],
    )
    lowered = [keyword.lower() for keyword in keywords]

    assert "ai" in lowered
    assert "seo" in lowered
    assert "linkedin" in lowered
    assert "marketing" in lowered


def test_keyword_ordering_is_deterministic_and_duplicates_removed() -> None:
    query = "AI AI SEO linkedin marketing linkedin strategy"
    sources = [
        {"title": "AI SEO marketing strategy for LinkedIn"},
        {"title": "LinkedIn AI strategy"},
    ]
    first = research_agent_module._build_keywords(query=query, sources=sources)  # noqa: SLF001
    second = research_agent_module._build_keywords(query=query, sources=sources)  # noqa: SLF001

    assert first == second
    assert first.count("ai") == 1
    assert first.count("linkedin") == 1
    assert first.count("seo") == 1
    assert first.index("ai") < first.index("seo") < first.index("linkedin")


def test_keyword_floor_stays_respected_when_query_is_all_stopwords() -> None:
    keywords = research_agent_module._build_keywords(  # noqa: SLF001
        query="the and of for to in on with about this that",
        sources=[],
    )

    assert len(keywords) >= 3
    assert all(isinstance(item, str) and item.strip() for item in keywords)
