from __future__ import annotations

from contentblitz.agents import blog_writer as blog_writer_module
from contentblitz.agents import linkedin_writer as linkedin_writer_module
from contentblitz.core.cost_controls import (
    normalize_cost_controls,
    preferred_text_model,
)
from contentblitz.core.model_policy import (
    KNOWN_TEXT_MODEL_AGENT_KEYS,
    build_text_model_policy,
    resolve_text_model,
)
from contentblitz.state import create_initial_state


def test_every_known_agent_key_resolves_to_model() -> None:
    policy = build_text_model_policy()
    for agent_key in KNOWN_TEXT_MODEL_AGENT_KEYS:
        resolved_default = resolve_text_model(
            agent_key,
            near_budget=False,
            policy=policy,
        )
        resolved_fallback = resolve_text_model(
            agent_key,
            near_budget=True,
            policy=policy,
        )
        assert resolved_default
        assert resolved_fallback


def test_unknown_agent_key_uses_safe_fallback() -> None:
    policy = build_text_model_policy()
    resolved_default = resolve_text_model("unknown_new_agent", near_budget=False)
    resolved_fallback = resolve_text_model("unknown_new_agent", near_budget=True)

    assert resolved_default == policy["unknown"].default_model
    assert resolved_fallback == policy["unknown"].fallback_model


def test_near_budget_selects_cheaper_research_model() -> None:
    normal_controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 100,
            "token_budget_per_session": 1000,
        }
    )
    near_budget_controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 950,
            "token_budget_per_session": 1000,
        }
    )

    assert preferred_text_model(
        normal_controls,
        agent_key="research_agent",
    ) == "gpt-4o"
    assert preferred_text_model(
        near_budget_controls,
        agent_key="research_agent",
    ) == "gpt-4o-mini"


def test_env_override_supports_claude_model_names(monkeypatch) -> None:
    monkeypatch.setenv(
        "CONTENTBLITZ_TEXT_MODEL_BLOG_WRITER_DEFAULT",
        "claude-3-5-haiku-20241022",
    )
    monkeypatch.setenv(
        "CONTENTBLITZ_TEXT_MODEL_BLOG_WRITER_FALLBACK",
        "claude-3-5-sonnet-20241022",
    )
    controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 0,
            "token_budget_per_session": 1000,
        }
    )
    near_budget_controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 900,
            "token_budget_per_session": 1000,
        }
    )

    assert preferred_text_model(controls, agent_key="blog_writer") == (
        "claude-3-5-haiku-20241022"
    )
    assert preferred_text_model(near_budget_controls, agent_key="blog_writer") == (
        "claude-3-5-sonnet-20241022"
    )


def test_invalid_env_override_fails_safely(monkeypatch) -> None:
    monkeypatch.setenv(
        "CONTENTBLITZ_TEXT_MODEL_QUERY_HANDLER_DEFAULT",
        "bad model name with spaces",
    )
    monkeypatch.setenv(
        "CONTENTBLITZ_TEXT_MODEL_QUERY_HANDLER_FALLBACK",
        "",
    )
    controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 0,
            "token_budget_per_session": 1000,
        }
    )

    assert preferred_text_model(controls, agent_key="query_handler") == "gpt-4o-mini"


def test_global_env_default_overrides_agent_defaults(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_TEXT_MODEL_DEFAULT", "gpt-5.4")
    monkeypatch.setenv("CONTENTBLITZ_TEXT_MODEL_FALLBACK", "gpt-5.4-mini")

    normal_controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 10,
            "token_budget_per_session": 1000,
        }
    )
    near_budget_controls = normalize_cost_controls(
        {
            "tokens_used_this_session": 950,
            "token_budget_per_session": 1000,
        }
    )

    assert preferred_text_model(
        normal_controls,
        agent_key="query_handler",
    ) == "gpt-5.4"
    assert preferred_text_model(
        near_budget_controls,
        agent_key="query_handler",
    ) == "gpt-5.4-mini"


def test_blog_writer_uses_model_policy(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_preferred_text_model(cost_controls, *, agent_key=None):
        _ = cost_controls
        seen["policy_agent_key"] = str(agent_key)
        return "policy-blog-model"

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (prompt, agent_key, metadata)
        seen["requested_model"] = model
        return {"output": "Blog draft content."}

    monkeypatch.setattr(
        blog_writer_module,
        "preferred_text_model",
        fake_preferred_text_model,
    )
    monkeypatch.setattr(blog_writer_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        user_query="Write a blog post about observability",
        content_brief={"blog": {}, "linkedin": {}, "image": {}},
    )

    _ = blog_writer_module.blog_writer_node(state)

    assert seen["policy_agent_key"] == "blog_writer"
    assert seen["requested_model"] == "policy-blog-model"


def test_linkedin_writer_uses_model_policy(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def fake_preferred_text_model(cost_controls, *, agent_key=None):
        _ = cost_controls
        seen["policy_agent_key"] = str(agent_key)
        return "policy-linkedin-model"

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (prompt, agent_key, metadata)
        seen["requested_model"] = model
        return {"output": "x" * 1500}

    monkeypatch.setattr(
        linkedin_writer_module,
        "preferred_text_model",
        fake_preferred_text_model,
    )
    monkeypatch.setattr(linkedin_writer_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        user_query="Write a LinkedIn post about observability",
        content_brief={"blog": {}, "linkedin": {}, "image": {}},
    )

    _ = linkedin_writer_module.linkedin_writer_node(state)

    assert seen["policy_agent_key"] == "linkedin_writer"
    assert seen["requested_model"] == "policy-linkedin-model"
