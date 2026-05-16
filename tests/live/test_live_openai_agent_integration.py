# tests/live/test_live_openai_agent_integration.py

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from contentblitz.state import create_initial_state
from contentblitz.agents.query_handler import query_handler_node
from contentblitz.agents.content_strategist import content_strategist_node
from contentblitz.agents.blog_writer import blog_writer_node


pytestmark = pytest.mark.skipif(
    os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") != "1",
    reason="Live OpenAI agent integration tests are disabled by default.",
)


def require_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")


def test_live_query_handler_uses_openai():
    require_openai_key()

    state = create_initial_state(
        user_query="Write a short LinkedIn post about AI content workflows."
    )

    result = query_handler_node(state)

    assert result
    assert result.get("requested_outputs")
    assert result.get("clarification_needed") is False
    assert result.get("cost_controls", {}).get("tokens_used_this_session", 0) >= 0


def test_live_content_strategist_uses_openai():
    require_openai_key()

    state = create_initial_state(
        user_query="Write a short blog post about AI content workflows."
    )

    state["requested_outputs"] = ["blog"]
    state["intent"] = "content_generation"
    state["research_required"] = False
    state["research_data"] = {
        "summary": "AI content workflows help teams plan, draft, review, and repurpose content faster.",
        "key_points": [
            "AI can reduce repetitive drafting work.",
            "Human review remains important.",
            "Workflow design matters more than tool choice.",
        ],
    }
    state["sources"] = []

    result = content_strategist_node(state)

    assert result
    assert "content_brief" in result
    assert result["content_brief"]["blog"]
    assert result.get("cost_controls", {}).get("tokens_used_this_session", 0) >= 0


def test_live_blog_writer_uses_openai():
    require_openai_key()

    state = create_initial_state(
        user_query="Write a short blog post about AI content workflows."
    )

    state["requested_outputs"] = ["blog"]
    state["intent"] = "content_generation"
    state["content_brief"]["blog"] = {
        "topic": "AI content workflows",
        "audience": "marketing teams",
        "angle": "practical workflow improvements",
        "key_points": [
            "Plan before generating.",
            "Use AI for drafting and repurposing.",
            "Keep human review in the loop.",
        ],
        "seo_keywords": ["AI content workflows", "content automation"],
    }
    state["sources"] = []

    result = blog_writer_node(state)

    assert result
    assert "content_drafts" in result
    assert result["content_drafts"]["blog"]["body"]
    assert result["content_drafts"]["blog"]["version"] >= 1
    assert result.get("cost_controls", {}).get("tokens_used_this_session", 0) >= 0
