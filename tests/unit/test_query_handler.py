import json

from contentblitz.agents import query_handler as query_handler_module
from contentblitz.state import create_initial_state


def _mock_llm(monkeypatch, payload):
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        assert agent_key == "query_handler"
        return {"output": payload}

    monkeypatch.setattr(query_handler_module, "generate_text", fake_generate_text)


def test_vague_query_triggers_clarification(monkeypatch) -> None:
    _mock_llm(monkeypatch, "not-json")
    state = create_initial_state(user_query="help")
    updates = query_handler_module.query_handler_node(state)

    assert updates["clarification_needed"] is True
    assert isinstance(updates["clarification_message"], str)
    assert updates["routing_decision"] == "clarification_node"


def test_image_query_sets_research_required_false(monkeypatch) -> None:
    payload = json.dumps(
        {
            "intent": "image_generation",
            "requested_outputs": ["image"],
            "research_required": True,
            "clarification_needed": False,
            "clarification_message": None,
            "export_requested": False,
        }
    )
    _mock_llm(monkeypatch, payload)
    state = create_initial_state(user_query="Create an image about cloud security")
    updates = query_handler_module.query_handler_node(state)

    assert updates["requested_outputs"] == ["image"]
    assert updates["research_required"] is False
    assert updates["routing_decision"] == "image_agent_node"


def test_blog_query_sets_research_required_true(monkeypatch) -> None:
    payload = json.dumps(
        {
            "intent": "content_creation",
            "requested_outputs": ["blog"],
            "research_required": True,
            "clarification_needed": False,
            "clarification_message": None,
            "export_requested": False,
        }
    )
    _mock_llm(monkeypatch, payload)
    state = create_initial_state(user_query="Write a blog post about AI trends")
    updates = query_handler_module.query_handler_node(state)

    assert updates["requested_outputs"] == ["blog"]
    assert updates["research_required"] is True
    assert updates["routing_decision"] == "research_agent_node"


def test_research_only_query_sets_requested_outputs(monkeypatch) -> None:
    _mock_llm(monkeypatch, "malformed")
    state = create_initial_state(user_query="Research EV charging adoption in 2026")
    updates = query_handler_module.query_handler_node(state)

    assert updates["requested_outputs"] == ["research"]
    assert updates["routing_decision"] == "research_agent_node"


def test_export_request_sets_export_requested_true(monkeypatch) -> None:
    _mock_llm(monkeypatch, "malformed")
    state = create_initial_state(user_query="Create a blog and export as PDF")
    updates = query_handler_module.query_handler_node(state)

    assert updates["export_requested"] is True


def test_budget_exceeded_routes_to_error_handler_node(monkeypatch) -> None:
    _mock_llm(monkeypatch, "malformed")
    state = create_initial_state(
        user_query="Write a blog post",
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": True,
        },
    )
    updates = query_handler_module.query_handler_node(state)

    assert updates["routing_decision"] == "error_handler_node"
    assert "errors" in updates
    assert len(updates["errors"]) == 1
    assert updates["errors"][0]["type"] == "budget_exceeded"


def test_malformed_llm_response_falls_back_safely(monkeypatch) -> None:
    _mock_llm(monkeypatch, "{not-valid-json")
    state = create_initial_state(user_query="Write a blog post on cybersecurity")
    updates = query_handler_module.query_handler_node(state)

    assert updates["intent"] in {"content_creation", "research", "image_generation", "clarification"}
    assert isinstance(updates["requested_outputs"], list)
    assert updates["clarification_needed"] is False
    assert updates["routing_decision"] in {
        "clarification_node",
        "image_agent_node",
        "research_agent_node",
        "content_strategist_node",
    }


def test_node_returns_only_state_updates(monkeypatch) -> None:
    _mock_llm(monkeypatch, "malformed")
    state = create_initial_state(user_query="Write a blog post on observability")
    updates = query_handler_module.query_handler_node(state)

    assert "session_id" not in updates
    assert "user_id" not in updates
    assert "conversation_history" not in updates


def test_query_handler_does_not_propagate_stale_retry_flags(monkeypatch) -> None:
    payload = json.dumps(
        {
            "intent": "content_creation",
            "requested_outputs": ["blog"],
            "research_required": True,
            "clarification_needed": False,
            "clarification_message": None,
            "export_requested": False,
        }
    )
    _mock_llm(monkeypatch, payload)
    state = create_initial_state(
        user_query="Write a blog post about AI workflows",
        retry_requested=True,
        retry_target="blog",
    )
    updates = query_handler_module.query_handler_node(state)
    assert "retry_requested" not in updates
    assert "retry_target" not in updates


def test_explicit_image_only_request_with_non_empty_query_routes_deterministically(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM classification should not run for explicit image-only UI requests.")

    monkeypatch.setattr(query_handler_module, "generate_text", fail_if_called)

    state = create_initial_state(
        user_query="create futuristic cyberpunk hoodie artwork for streetwear branding",
        requested_outputs=["image"],
        research_required=False,
        export_requested=False,
    )
    updates = query_handler_module.query_handler_node(state)

    assert updates["requested_outputs"] == ["image"]
    assert updates["research_required"] is False
    assert updates["routing_decision"] == "image_agent_node"
