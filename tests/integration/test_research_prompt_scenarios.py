from typing import Any

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph


def _run_prompt(prompt: str) -> dict[str, Any]:
    state = create_initial_state(user_query=prompt)
    graph = build_langgraph()
    return graph.invoke(state)


def _draft_body(result: dict[str, Any], output_type: str) -> str:
    content_drafts = result.get("content_drafts") or {}
    draft = content_drafts.get(output_type) or {}
    return draft.get("body") or ""


def _errors_are_nonfatal(result: dict[str, Any]) -> bool:
    errors = result.get("errors") or []
    return all(error.get("recoverable") is True for error in errors)


def _assert_research_populated(result: dict[str, Any]) -> None:
    research_data = result.get("research_data") or {}

    assert result["research_required"] is True
    assert research_data
    assert research_data.get("synthesized_summary")
    assert research_data.get("quality") in {"standard", "degraded"}
    assert len(research_data.get("key_facts") or []) >= 3
    assert len(research_data.get("keywords") or []) >= 3
    assert len(result.get("sources") or []) >= 1


def _assert_no_text_drafts(result: dict[str, Any]) -> None:
    assert _draft_body(result, "blog") == ""
    assert _draft_body(result, "linkedin") == ""


def _assert_degraded_research_not_cached(result: dict[str, Any]) -> None:
    research_data = result.get("research_data") or {}
    cache_metadata = result.get("cache_metadata") or {}

    if research_data.get("quality") == "degraded":
        assert cache_metadata.get("keys") == []


def test_standard_research_prompt_populates_research_data() -> None:
    result = _run_prompt("research AI content marketing trends for 2026")

    assert result["requested_outputs"] == ["research"]
    assert result["routing_decision"] == "research_agent_node"
    _assert_research_populated(result)
    _assert_no_text_drafts(result)
    _assert_degraded_research_not_cached(result)
    assert result["quality_scores"] == {}
    assert _errors_are_nonfatal(result)


def test_multi_topic_research_prompt_populates_keywords_and_sources() -> None:
    result = _run_prompt(
        "research the impact of AI agents on ecommerce, SEO, and social media marketing"
    )

    assert result["requested_outputs"] == ["research"]
    _assert_research_populated(result)

    keywords = result["research_data"].get("keywords") or []
    assert any(keyword in keywords for keyword in ["ecommerce", "seo", "marketing"])

    _assert_no_text_drafts(result)
    _assert_degraded_research_not_cached(result)
    assert _errors_are_nonfatal(result)


def test_obscure_research_prompt_uses_degraded_fallback_safely() -> None:
    result = _run_prompt("research obscure synthetic biology fashion fabrics")

    assert result["requested_outputs"] == ["research"]
    _assert_research_populated(result)
    assert result["research_data"]["quality"] in {"degraded", "standard"}

    if result["research_data"]["quality"] == "degraded":
        assert result["cache_metadata"]["keys"] == []

    _assert_no_text_drafts(result)
    assert _errors_are_nonfatal(result)


def test_large_multi_domain_research_prompt_respects_cost_controls() -> None:
    result = _run_prompt(
        "research AI trends in fintech, healthcare, ecommerce, logistics, and education"
    )

    assert result["requested_outputs"] == ["research"]
    _assert_research_populated(result)

    cost_controls = result.get("cost_controls") or {}
    assert cost_controls.get("search_queries_used_this_session", 0) >= 1
    assert cost_controls.get("budget_exceeded") is False

    _assert_no_text_drafts(result)
    _assert_degraded_research_not_cached(result)
    assert _errors_are_nonfatal(result)


def test_short_ai_trends_prompt_routes_to_clarification() -> None:
    result = _run_prompt("AI trends")

    assert result["clarification_needed"] is True or result["intent"] == "clarification"
    assert result["research_required"] is False
    assert result.get("research_data") == {}
    assert len(result.get("sources") or []) == 0
    _assert_no_text_drafts(result)
    assert result["quality_scores"] == {}
    assert _errors_are_nonfatal(result)


def test_short_marketing_prompt_routes_to_clarification() -> None:
    result = _run_prompt("marketing")

    assert result["clarification_needed"] is True
    assert result["routing_decision"] == "clarification_node"
    assert result["research_required"] is False
    assert result.get("research_data") == {}
    assert len(result.get("sources") or []) == 0
    _assert_no_text_drafts(result)
    assert result["quality_scores"] == {}
    assert _errors_are_nonfatal(result)


def test_garbage_research_like_input_does_not_crash() -> None:
    result = _run_prompt("asdfasdfasdf")

    assert result["clarification_needed"] is True or result["workflow_status"] != "failed"
    assert _errors_are_nonfatal(result)

    if result["clarification_needed"] is True:
        assert result.get("research_data") == {}
        assert len(result.get("sources") or []) == 0
        _assert_no_text_drafts(result)
        assert result["quality_scores"] == {}


def test_research_prompt_does_not_create_content_briefs() -> None:
    result = _run_prompt("research AI adoption trends in ecommerce marketing")

    assert result["requested_outputs"] == ["research"]
    _assert_research_populated(result)

    content_brief = result.get("content_brief") or {}
    assert content_brief.get("blog") == {}
    assert content_brief.get("linkedin") == {}
    assert content_brief.get("image") == {}

    _assert_no_text_drafts(result)
    assert result["quality_scores"] == {}
    assert _errors_are_nonfatal(result)


def test_research_sources_have_required_fields() -> None:
    result = _run_prompt("research AI content marketing trends for 2026")

    _assert_research_populated(result)

    for source in result.get("sources") or []:
        assert "title" in source
        assert "provider" in source
        assert "snippet" in source
        assert "credibility_score" in source
        assert "citation_available" in source

        if source.get("url") is None:
            assert source["citation_available"] is False


def test_research_cost_controls_do_not_affect_other_counters() -> None:
    result = _run_prompt("research the future of AI marketing systems")

    _assert_research_populated(result)

    cost_controls = result.get("cost_controls") or {}
    assert cost_controls.get("search_queries_used_this_session", 0) >= 1
    assert cost_controls.get("image_generations_used_this_session") == 0
    assert cost_controls.get("total_retries_used_this_session") == 0