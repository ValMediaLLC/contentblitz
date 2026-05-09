from contentblitz.core.cache_keys import (
    build_research_cache_key,
    normalize_query,
    sha256_normalized_query,
)


def test_normalized_equivalent_queries_produce_same_key() -> None:
    q1 = "  AI   Content   Workflow  "
    q2 = "ai content workflow"
    assert normalize_query(q1) == normalize_query(q2)
    assert build_research_cache_key(q1, depth="standard") == build_research_cache_key(
        q2, depth="standard"
    )


def test_raw_user_input_not_visible_in_cache_key() -> None:
    query = "Highly Specific User Query 2026!"
    key = build_research_cache_key(query, depth="standard")
    assert query not in key
    assert "Highly" not in key
    assert "Specific" not in key
    assert key.startswith("research:")
    assert len(sha256_normalized_query(query)) == 64


def test_different_depth_values_produce_different_keys() -> None:
    query = "ai content strategy"
    standard_key = build_research_cache_key(query, depth="standard")
    deep_key = build_research_cache_key(query, depth="deep")
    fallback_key = build_research_cache_key(query, depth="fallback")

    assert standard_key != deep_key
    assert standard_key != fallback_key
    assert deep_key != fallback_key
    assert standard_key.endswith(":standard")
    assert deep_key.endswith(":deep")
    assert fallback_key.endswith(":fallback")
