import json

from contentblitz.agents import content_strategist as strategist_module
from contentblitz.state import create_initial_state


def test_blog_request_creates_blog_brief(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        assert agent_key == "content_strategist"
        return {"output": json.dumps({"format": "blog", "angle": "deep dive", "objective": "educate"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(requested_outputs=["blog"], user_query="B2B SaaS onboarding", intent="content_creation")
    updates = strategist_module.content_strategist_node(state)

    assert updates["content_brief"]["blog"]["format"] == "blog"
    assert updates["content_brief"]["blog"]["angle"] == "deep dive"
    assert "content_drafts" not in updates


def test_linkedin_request_creates_linkedin_brief(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "linkedin", "structure": ["hook", "insight", "cta"]})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(requested_outputs=["linkedin"], user_query="remote collaboration", intent="content_creation")
    updates = strategist_module.content_strategist_node(state)

    assert updates["content_brief"]["linkedin"]["format"] == "linkedin"
    assert "structure" in updates["content_brief"]["linkedin"]
    assert "content_drafts" not in updates


def test_image_request_creates_image_brief(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "image", "visual_direction": "neo-futurist"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(requested_outputs=["image"], user_query="fintech dashboard concept", intent="image_generation")
    updates = strategist_module.content_strategist_node(state)

    assert updates["content_brief"]["image"]["format"] == "image"
    assert updates["content_brief"]["image"]["visual_direction"] == "neo-futurist"


def test_research_output_creates_research_report(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "blog", "angle": "analysis-led"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["research", "blog"],
        user_query="AI governance patterns",
        research_data={"synthesized_summary": "Governance models are converging."},
        sources=[{"title": "Source A", "url": "https://example.com/a", "snippet": "long snippet"}],
    )
    original_blog_draft = dict(state["content_drafts"]["blog"])
    original_linkedin_draft = dict(state["content_drafts"]["linkedin"])
    updates = strategist_module.content_strategist_node(state)

    assert "content_drafts" in updates
    assert updates["content_drafts"]["research_report"]["title"]
    assert updates["content_drafts"]["research_report"]["body"]
    assert isinstance(updates["content_drafts"]["research_report"]["sections"], list)
    assert len(updates["content_drafts"]["research_report"]["sections"]) >= 1
    assert updates["content_drafts"]["blog"] == original_blog_draft
    assert updates["content_drafts"]["linkedin"] == original_linkedin_draft


def test_malformed_json_falls_back_to_deterministic_brief(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "{bad-json"}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["blog"],
        user_query="customer retention strategy",
        intent="content_creation",
        brand_voice={"tone": "confident"},
        research_data={"summary": "Retention benchmarks improved year-over-year."},
    )
    updates = strategist_module.content_strategist_node(state)
    brief = updates["content_brief"]["blog"]

    assert brief["format"] == "blog"
    assert brief["objective"]
    assert brief["tone"] == "confident"


def test_token_counter_increments_after_generate_text_result(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {
            "output": json.dumps({"format": "linkedin", "angle": "short-form insight"}),
            "usage": {"total_tokens": 42},
        }

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["linkedin"],
        user_query="team productivity",
        cost_controls={
            "tokens_used_this_session": 10,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
        },
    )
    updates = strategist_module.content_strategist_node(state)

    assert updates["cost_controls"]["tokens_used_this_session"] == 52


def test_blog_only_does_not_populate_research_report(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "blog", "angle": "educational"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(requested_outputs=["blog"], user_query="future ai workflows")
    updates = strategist_module.content_strategist_node(state)

    assert "content_drafts" not in updates


def test_linkedin_only_does_not_populate_research_report(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "linkedin", "angle": "trend memo"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(requested_outputs=["linkedin"], user_query="ai content marketing trends")
    updates = strategist_module.content_strategist_node(state)

    assert "content_drafts" not in updates


def test_content_drafts_blog_and_linkedin_not_modified_by_content_strategist(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "blog", "angle": "insight-led"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["research", "blog"],
        user_query="ai workflow adoption",
        content_drafts={
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        research_data={"synthesized_summary": "Adoption is increasing."},
    )
    original_blog_draft = dict(state["content_drafts"]["blog"])
    original_linkedin_draft = dict(state["content_drafts"]["linkedin"])

    updates = strategist_module.content_strategist_node(state)
    assert updates["content_drafts"]["blog"] == original_blog_draft
    assert updates["content_drafts"]["linkedin"] == original_linkedin_draft
