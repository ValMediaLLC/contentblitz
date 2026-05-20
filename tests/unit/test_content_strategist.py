import asyncio
import json
import threading
import time

from contentblitz.agents import content_strategist as strategist_module
from contentblitz.state import create_initial_state


def _output_type_from_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    for output_type in ("blog", "linkedin", "image"):
        if f"'{output_type}'" in lowered:
            return output_type
    return "unknown"


def test_blog_request_creates_blog_brief(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        assert agent_key == "content_strategist"
        return {
            "output": json.dumps(
                {"format": "blog", "angle": "deep dive", "objective": "educate"}
            )
        }

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["blog"],
        user_query="B2B SaaS onboarding",
        intent="content_creation",
    )
    updates = strategist_module.content_strategist_node(state)

    assert updates["content_brief"]["blog"]["format"] == "blog"
    assert updates["content_brief"]["blog"]["angle"] == "deep dive"
    assert "content_drafts" not in updates


def test_linkedin_request_creates_linkedin_brief(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {
            "output": json.dumps(
                {"format": "linkedin", "structure": ["hook", "insight", "cta"]}
            )
        }

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["linkedin"],
        user_query="remote collaboration",
        intent="content_creation",
    )
    updates = strategist_module.content_strategist_node(state)

    assert updates["content_brief"]["linkedin"]["format"] == "linkedin"
    assert "structure" in updates["content_brief"]["linkedin"]
    assert "content_drafts" not in updates


def test_image_request_creates_image_brief(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {
            "output": json.dumps(
                {"format": "image", "visual_direction": "neo-futurist"}
            )
        }

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["image"],
        user_query="fintech dashboard concept",
        intent="image_generation",
    )
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
        sources=[
            {
                "title": "Source A",
                "url": "https://example.com/a",
                "snippet": "long snippet",
            }
        ],
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


def test_async_fanout_populates_all_requested_briefs(monkeypatch) -> None:
    inflight = {"current": 0, "max": 0}
    lock = threading.Lock()

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (model, metadata)
        assert agent_key == "content_strategist"
        output_type = _output_type_from_prompt(prompt)
        with lock:
            inflight["current"] += 1
            inflight["max"] = max(inflight["max"], inflight["current"])
        time.sleep(0.02)
        with lock:
            inflight["current"] -= 1
        if output_type == "blog":
            return {"output": json.dumps({"format": "blog", "angle": "deep dive"})}
        if output_type == "linkedin":
            return {
                "output": json.dumps(
                    {"format": "linkedin", "structure": ["hook", "insight", "cta"]}
                )
            }
        return {
            "output": json.dumps({"format": "image", "visual_direction": "modern"})
        }

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["image", "blog", "linkedin"],
        user_query="best electric cars in 2026",
        intent="content_creation",
    )
    updates = strategist_module.content_strategist_node(state)

    assert inflight["max"] >= 2
    assert updates["content_brief"]["blog"]["format"] == "blog"
    assert updates["content_brief"]["linkedin"]["format"] == "linkedin"
    assert updates["content_brief"]["image"]["format"] == "image"


def test_token_updates_applied_in_deterministic_output_order(monkeypatch) -> None:
    apply_order: list[str] = []

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (model, metadata)
        assert agent_key == "content_strategist"
        output_type = _output_type_from_prompt(prompt)
        if output_type == "blog":
            time.sleep(0.03)
        if output_type == "linkedin":
            time.sleep(0.01)
        return {
            "output": json.dumps({"format": output_type, "angle": f"{output_type}-a"}),
            "marker": output_type,
        }

    def fake_apply_text_tokens(cost_controls, llm_response):
        marker = str(llm_response.get("marker", "")).strip().lower()
        if marker:
            apply_order.append(marker)
        updated = dict(cost_controls)
        updated["tokens_used_this_session"] = (
            int(updated.get("tokens_used_this_session", 0)) + 1
        )
        return updated

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(strategist_module, "apply_text_tokens", fake_apply_text_tokens)

    state = create_initial_state(
        requested_outputs=["image", "linkedin", "blog"],
        user_query="autonomous freight trends",
    )
    updates = strategist_module.content_strategist_node(state)

    assert apply_order == ["blog", "linkedin", "image"]
    assert updates["cost_controls"]["tokens_used_this_session"] == 3


def test_failed_output_falls_back_without_blocking_other_briefs(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (model, metadata)
        assert agent_key == "content_strategist"
        output_type = _output_type_from_prompt(prompt)
        if output_type == "linkedin":
            raise RuntimeError("provider unavailable")
        if output_type == "blog":
            return {"output": json.dumps({"format": "blog", "angle": "evidence-led"})}
        return {
            "output": json.dumps({"format": "image", "visual_direction": "cinematic"})
        }

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["blog", "linkedin", "image"],
        user_query="grid-scale batteries",
        intent="content_creation",
    )
    updates = strategist_module.content_strategist_node(state)

    assert updates["content_brief"]["blog"]["angle"] == "evidence-led"
    assert updates["content_brief"]["image"]["visual_direction"] == "cinematic"
    linkedin = updates["content_brief"]["linkedin"]
    assert linkedin["format"] == "linkedin"
    assert linkedin["objective"].startswith("Requested deliverable:")


def test_content_strategist_fanout_handles_running_event_loop(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (model, metadata)
        assert agent_key == "content_strategist"
        output_type = _output_type_from_prompt(prompt)
        return {"output": json.dumps({"format": output_type, "angle": output_type})}

    def fail_asyncio_run(*args, **kwargs):
        raise AssertionError("asyncio.run should not be called inside a running loop")

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(strategist_module.asyncio, "run", fail_asyncio_run)

    state = create_initial_state(
        requested_outputs=["blog", "linkedin", "image"],
        user_query="async loop compatibility",
        intent="content_creation",
    )

    async def _invoke_node():
        return strategist_module.content_strategist_node(state)

    event_loop = asyncio.new_event_loop()
    try:
        updates = event_loop.run_until_complete(_invoke_node())
    finally:
        event_loop.close()

    assert updates["content_brief"]["blog"]["format"] == "blog"
    assert updates["content_brief"]["linkedin"]["format"] == "linkedin"
    assert updates["content_brief"]["image"]["format"] == "image"


def test_provider_latency_metadata_recorded_for_content_strategist_calls(
    monkeypatch,
) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        _ = (prompt, metadata)
        assert agent_key == "content_strategist"
        assert model
        time.sleep(0.002)
        return {
            "output": json.dumps(
                {"format": "linkedin", "structure": ["hook", "insight", "cta"]}
            ),
            "provider": "openai",
            "model": "gpt-4o",
        }

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["linkedin"],
        user_query="remote collaboration",
        intent="content_creation",
    )
    updates = strategist_module.content_strategist_node(state)
    strategist_metrics = updates["tool_outputs"]["content_strategist"]

    assert strategist_metrics["provider"] == "openai"
    assert strategist_metrics["model"] == "gpt-4o"
    assert strategist_metrics["provider_call_count"] == 1
    assert strategist_metrics["provider_call_count_by_output_type"]["linkedin"] == 1
    assert strategist_metrics["provider_latency_total_ms"] >= 0
    assert strategist_metrics["provider_latency_wall_ms"] >= 0
    assert strategist_metrics["provider_latency_by_output_type_ms"]["linkedin"] >= 0
    assert isinstance(strategist_metrics["provider_latency_ms"], int)
    assert strategist_metrics["provider_latency_ms"] >= 0
    assert (
        strategist_metrics["provider_latency_ms"]
        == strategist_metrics["provider_latency_wall_ms"]
    )


def test_budget_fallback_without_provider_call_omits_latency_metadata(
    monkeypatch,
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "generate_text should not run when token budget is exhausted"
        )

    monkeypatch.setattr(strategist_module, "generate_text", fail_if_called)
    state = create_initial_state(
        requested_outputs=["blog"],
        user_query="future ai workflows",
        cost_controls={
            "tokens_used_this_session": 10000,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "token_budget_per_session": 10000,
            "search_query_cap_per_session": 5,
            "image_generation_cap_per_session": 3,
            "max_total_retries_per_session": 3,
            "budget_exceeded": True,
        },
    )

    updates = strategist_module.content_strategist_node(state)

    assert "tool_outputs" not in updates or "content_strategist" not in updates.get(
        "tool_outputs", {}
    )


def test_blog_only_does_not_populate_research_report(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "blog", "angle": "educational"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["blog"], user_query="future ai workflows"
    )
    updates = strategist_module.content_strategist_node(state)

    assert "content_drafts" not in updates


def test_linkedin_only_does_not_populate_research_report(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": json.dumps({"format": "linkedin", "angle": "trend memo"})}

    monkeypatch.setattr(strategist_module, "generate_text", fake_generate_text)
    state = create_initial_state(
        requested_outputs=["linkedin"], user_query="ai content marketing trends"
    )
    updates = strategist_module.content_strategist_node(state)

    assert "content_drafts" not in updates


def test_content_drafts_blog_and_linkedin_not_modified_by_content_strategist(
    monkeypatch,
) -> None:
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
