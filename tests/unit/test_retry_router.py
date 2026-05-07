from contentblitz.agents import blog_writer as blog_writer_module
from contentblitz.agents import linkedin_writer as linkedin_writer_module
from contentblitz.agents import retry_router as retry_router_module
from contentblitz.state import create_initial_state


def test_retry_router_increments_total_retries_used_this_session() -> None:
    state = create_initial_state(
        retry_requested=True,
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 2,
            "budget_exceeded": False,
        },
    )
    updates = retry_router_module.retry_router_node(state)
    assert updates["cost_controls"]["total_retries_used_this_session"] == 3


def test_retry_router_no_retry_request_returns_no_updates() -> None:
    state = create_initial_state(retry_requested=False)
    updates = retry_router_module.retry_router_node(state)
    assert updates == {}


def test_writers_do_not_increment_total_retries_used_this_session(monkeypatch) -> None:
    def fake_blog_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Blog draft text."}

    def fake_linkedin_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Too short.\n\n#AI"}

    monkeypatch.setattr(blog_writer_module, "generate_text", fake_blog_generate_text)
    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_linkedin_generate_text)

    cost_controls = {
        "tokens_used_this_session": 0,
        "search_queries_used_this_session": 0,
        "image_generations_used_this_session": 0,
        "total_retries_used_this_session": 0,
        "budget_exceeded": False,
    }

    blog_state = create_initial_state(
        user_query="AI workflows in marketing",
        content_brief={
            "blog": {
                "objective": "Educate teams.",
                "audience": "marketing leaders",
                "tone": "practical",
                "angle": "playbook",
                "outline": ["Problem", "Approach", "Steps"],
            },
            "linkedin": {},
            "image": {},
        },
        cost_controls=cost_controls,
    )
    blog_updates = blog_writer_module.blog_writer_node(blog_state)
    assert blog_updates["cost_controls"]["total_retries_used_this_session"] == 0

    linkedin_state = create_initial_state(
        user_query="AI workflow systems for marketing teams",
        content_brief={
            "blog": {},
            "linkedin": {
                "objective": "Help teams operationalize AI content systems.",
                "audience": "marketing leaders",
                "tone": "direct and practical",
                "angle": "operational playbook",
                "structure": ["Hook", "Insight", "Framework", "CTA"],
            },
            "image": {},
        },
        cost_controls=cost_controls,
    )
    linkedin_updates = linkedin_writer_module.linkedin_writer_node(linkedin_state)
    assert linkedin_updates["cost_controls"]["total_retries_used_this_session"] == 0
