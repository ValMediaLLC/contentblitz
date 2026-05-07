from contentblitz.agents import blog_writer as blog_writer_module
from contentblitz.state import create_initial_state


def _base_state(**overrides):
    state = create_initial_state(
        user_query="AI workflows in marketing",
        content_brief={
            "blog": {
                "objective": "Educate marketing teams on AI workflow patterns.",
                "audience": "marketing leaders",
                "tone": "clear and practical",
                "angle": "seo playbook",
                "outline": ["Problem", "Framework", "Execution"],
            },
            "linkedin": {},
            "image": {},
        },
    )
    state.update(overrides)
    return state


def test_version_starts_at_1(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Draft body content."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    updates = blog_writer_module.blog_writer_node(_base_state())
    assert updates["content_drafts"]["blog"]["version"] == 1


def test_retry_increments_version(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Retry draft content."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        content_drafts={
            "blog": {"body": "Old", "version": 1},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        }
    )
    updates = blog_writer_module.blog_writer_node(state)
    assert updates["content_drafts"]["blog"]["version"] == 2


def test_citations_included_when_available(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Main draft."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        sources=[
            {
                "title": "Industry Report",
                "url": "https://example.com/report",
                "citation_available": True,
            }
        ]
    )
    updates = blog_writer_module.blog_writer_node(state)
    body = updates["content_drafts"]["blog"]["body"]
    assert "## Sources" in body
    assert "https://example.com/report" in body


def test_no_fake_citations_when_unavailable(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Main draft."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        sources=[
            {
                "title": "Unverified Source",
                "url": "https://example.com/unverified",
                "citation_available": False,
            }
        ]
    )
    updates = blog_writer_module.blog_writer_node(state)
    body = updates["content_drafts"]["blog"]["body"]
    assert "## Sources" not in body
    assert "https://example.com/unverified" not in body


def test_disclaimer_added_when_sources_unavailable(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Main draft."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    updates = blog_writer_module.blog_writer_node(_base_state(sources=[]))
    body = updates["content_drafts"]["blog"]["body"]
    assert "Disclaimer: No verifiable external citations were available" in body


def test_token_counter_increments(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Main draft.", "usage": {"total_tokens": 33}}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        cost_controls={
            "tokens_used_this_session": 10,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
        }
    )
    updates = blog_writer_module.blog_writer_node(state)
    assert updates["cost_controls"]["tokens_used_this_session"] == 43
    assert updates["cost_controls"]["total_retries_used_this_session"] == 0


def test_near_token_cap_uses_gpt4o_mini(monkeypatch) -> None:
    seen = {"model": None}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        seen["model"] = model
        return {"output": "Main draft."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        cost_controls={
            "tokens_used_this_session": 910,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        }
    )
    updates = blog_writer_module.blog_writer_node(state)
    assert seen["model"] == "gpt-4o-mini"
    assert updates["content_drafts"]["blog"]["model_used"] == "gpt-4o-mini"


def test_retry_feedback_included_in_prompt(monkeypatch) -> None:
    seen = {"prompt": ""}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        seen["prompt"] = prompt
        return {"output": "Main draft."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        retry_feedback={
            "blog": ["Improve SEO keyword density.", "Use a stronger CTA."],
            "linkedin": [],
        }
    )
    blog_writer_module.blog_writer_node(state)
    assert "Retry feedback to address" in seen["prompt"]
    assert "Improve SEO keyword density." in seen["prompt"]
    assert "Use a stronger CTA." in seen["prompt"]


def test_attempt_history_not_written_and_quality_not_scored(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Main draft."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    updates = blog_writer_module.blog_writer_node(_base_state())
    assert "attempt_history" not in updates
    assert "quality_scores" not in updates


def test_blog_writer_sets_draft_status_complete(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Main draft."}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    updates = blog_writer_module.blog_writer_node(_base_state())
    assert updates["draft_status"]["blog"] == "complete"
