from contentblitz.agents import linkedin_writer as linkedin_writer_module
from contentblitz.state import create_initial_state


def _base_state(**overrides):
    state = create_initial_state(
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
    )
    state.update(overrides)
    return state


def _long_linkedin_post() -> str:
    paragraph = (
        "Most teams are not blocked by ideas. They are blocked by repeatability. "
        "A single prompt can create momentum, but only systems create output you can trust week after week. "
        "The strongest teams define ownership, sequence, and measurable quality thresholds. "
    )
    body = (
        "If your AI content workflow still feels random, this is the fix.\n\n"
        + (paragraph * 5)
        + "\n\nWhat is one workflow stage you would standardize first? Share below.\n"
        "#AI #MarketingOps #ContentStrategy"
    )
    return body


def _very_long_linkedin_post() -> str:
    return (
        "If your AI content workflow still feels random, this is the fix.\n\n"
        + ("Most teams are not blocked by ideas. They are blocked by repeatability. " * 30)
        + "\n\nWhat is one workflow stage you would standardize first? Share below.\n"
        "#AI #MarketingOps #ContentStrategy"
    )


def test_version_starts_at_1(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": _long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert updates["content_drafts"]["linkedin"]["version"] == 1


def test_retry_increments_version(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": _long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        content_drafts={
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "old", "version": 1},
            "research_report": {"body": ""},
        }
    )
    updates = linkedin_writer_module.linkedin_writer_node(state)
    assert updates["content_drafts"]["linkedin"]["version"] == 2


def test_hook_extracted(monkeypatch) -> None:
    post = (
        "Stop treating AI as a writing shortcut.\n\n"
        + ("Operational discipline wins. " * 60)
        + "\n\nWhat process would you standardize first?\n"
        "#AI #Workflow #Marketing"
    )

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": post}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert updates["content_drafts"]["linkedin"]["hook"] == "Stop treating AI as a writing shortcut."


def test_cta_extracted(monkeypatch) -> None:
    post = (
        "AI execution is a systems challenge.\n\n"
        + ("Define constraints, then ship weekly. " * 60)
        + "\n\nWhat is one step your team can improve this week? Share below.\n"
        "#AI #MarketingOps #B2B"
    )

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": post}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert "share below" in updates["content_drafts"]["linkedin"]["cta"].lower()


def test_hashtags_extracted(monkeypatch) -> None:
    post = (
        "Build repeatable AI workflows.\n\n"
        + ("Use standards, not chaos. " * 70)
        + "\n\nWhat is your first implementation step?\n"
        "#AI #MarketingOps #LinkedInGrowth"
    )

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": post}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert updates["content_drafts"]["linkedin"]["hashtags"][:3] == [
        "#AI",
        "#MarketingOps",
        "#LinkedInGrowth",
    ]


def test_over_length_post_is_trimmed(monkeypatch) -> None:
    long_post = (
        "Lead with signal, not noise.\n\n"
        + ("This sentence adds detail to exceed the max range. " * 120)
        + "\n\nWhat part of your funnel needs the clearest process?\n"
        "#AI #ContentOps #DemandGen"
    )

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": long_post}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert len(updates["content_drafts"]["linkedin"]["body"]) <= 1600


def test_over_length_post_truncates_cleanly_and_preserves_tail(monkeypatch) -> None:
    source_post = _very_long_linkedin_post()
    cta = "What is one workflow stage you would standardize first? Share below."
    hashtags = "#AI #MarketingOps #ContentStrategy"

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": source_post}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    body = updates["content_drafts"]["linkedin"]["body"]

    assert len(body) <= 1600
    assert cta in body
    assert hashtags in body
    assert body.rstrip().endswith(hashtags)

    head = body.split(cta, 1)[0].rstrip()
    assert source_post.startswith(head)
    next_char = source_post[len(head):len(head) + 1]
    if next_char:
        assert next_char.isspace() or next_char in ".!?,;:"


def test_under_length_post_triggers_retry_behavior(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"output": "Too short.\n\n#AI"}
        return {"output": _long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert calls["count"] == 2
    assert "retry_counts" not in updates
    assert updates["content_drafts"]["linkedin"]["character_count"] >= 1300
    assert updates["cost_controls"]["total_retries_used_this_session"] == 0


def test_retry_feedback_is_included_in_prompt(monkeypatch) -> None:
    seen = {"prompt": ""}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        if not seen["prompt"]:
            seen["prompt"] = prompt
        return {"output": _long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        retry_feedback={
            "blog": [],
            "linkedin": ["Strengthen opening contrast.", "Use a sharper CTA."],
        }
    )
    linkedin_writer_module.linkedin_writer_node(state)

    assert "Retry feedback to address" in seen["prompt"]
    assert "Strengthen opening contrast." in seen["prompt"]
    assert "Use a sharper CTA." in seen["prompt"]


def test_token_counter_increments(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": _long_linkedin_post(), "usage": {"total_tokens": 27}}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        cost_controls={
            "tokens_used_this_session": 10,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
        }
    )
    updates = linkedin_writer_module.linkedin_writer_node(state)
    assert updates["cost_controls"]["tokens_used_this_session"] == 37
    assert updates["cost_controls"]["total_retries_used_this_session"] == 0


def test_near_token_cap_uses_gpt4o_mini(monkeypatch) -> None:
    seen = {"model": None}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        seen["model"] = model
        return {"output": _long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    state = _base_state(
        cost_controls={
            "tokens_used_this_session": 920,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        }
    )
    updates = linkedin_writer_module.linkedin_writer_node(state)
    assert seen["model"] == "gpt-4o-mini"
    assert updates["content_drafts"]["linkedin"]["model_used"] == "gpt-4o-mini"


def test_node_writes_only_allowed_fields(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": _long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert set(updates.keys()) == {"content_drafts", "draft_status", "cost_controls"}


def test_character_count_matches_final_body_length(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": _very_long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    body = updates["content_drafts"]["linkedin"]["body"]
    assert updates["content_drafts"]["linkedin"]["character_count"] == len(body)


def test_linkedin_writer_sets_draft_status_complete(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": _long_linkedin_post()}

    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    updates = linkedin_writer_module.linkedin_writer_node(_base_state())
    assert updates["draft_status"]["linkedin"] == "complete"
