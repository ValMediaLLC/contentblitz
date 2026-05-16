import importlib

from contentblitz.state import create_initial_state

clarification_node_module = importlib.import_module("contentblitz.agents.clarification")
clarification_wrapper_module = importlib.import_module(
    "contentblitz.agents.clarification_node"
)


def _base_state(**overrides):
    state = create_initial_state(user_query="blog")
    state.update(overrides)
    return state


def test_existing_clarification_message_is_used(monkeypatch) -> None:
    def should_not_be_called(prompt, agent_key, model="gpt-4o", metadata=None):
        raise AssertionError("generate_text should not be called when message exists")

    monkeypatch.setattr(
        clarification_node_module,
        "generate_text",
        should_not_be_called,
    )

    state = _base_state(clarification_message="Could you share your audience?")
    updates = clarification_node_module.clarification_node(state)

    assert updates["clarification_message"] == "Could you share your audience?"
    assert updates["final_response"] == "Could you share your audience?"
    assert updates["workflow_status"] == "awaiting_clarification"


def test_missing_message_triggers_generate_text(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        calls["count"] += 1
        assert agent_key == "query_handler"
        assert "User query: blog" in prompt
        return {"output": "Could you confirm the desired output format?"}

    monkeypatch.setattr(clarification_node_module, "generate_text", fake_generate_text)

    state = _base_state(clarification_message=None)
    updates = clarification_node_module.clarification_node(state)

    assert calls["count"] == 1
    assert (
        updates["clarification_message"]
        == "Could you confirm the desired output format?"
    )
    assert updates["final_response"] == "Could you confirm the desired output format?"


def test_generate_text_failure_uses_static_fallback(monkeypatch) -> None:
    def failing_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        raise RuntimeError("tool unavailable")

    monkeypatch.setattr(
        clarification_node_module,
        "generate_text",
        failing_generate_text,
    )

    state = _base_state(clarification_message=None)
    updates = clarification_node_module.clarification_node(state)

    assert (
        updates["clarification_message"]
        == "Could you clarify your goal, target audience, and desired output format?"
    )
    assert updates["final_response"] == updates["clarification_message"]


def test_conversation_history_appends_user_turn() -> None:
    state = _base_state(
        user_query="  marketing  ",
        clarification_message="Can you narrow the marketing goal?",
        conversation_history=[{"role": "assistant", "content": "How can I help?"}],
    )
    updates = clarification_node_module.clarification_node(state)

    assert len(updates["conversation_history"]) == 2
    assert updates["conversation_history"][0] == {
        "role": "assistant",
        "content": "How can I help?",
    }
    assert updates["conversation_history"][-1] == {
        "role": "user",
        "content": "marketing",
    }


def test_workflow_status_is_awaiting_clarification() -> None:
    state = _base_state(clarification_message="What format do you want?")
    updates = clarification_node_module.clarification_node(state)

    assert updates["workflow_status"] == "awaiting_clarification"
    assert updates["assembled_outputs"] == {}
    assert updates["export_outputs"] == {}
    assert "retry_counts" not in updates
    assert "cost_controls" not in updates
    assert "content_drafts" not in updates
    assert "quality_scores" not in updates


def test_compatibility_wrapper_exports_same_clarification_node() -> None:
    assert (
        clarification_wrapper_module.clarification_node
        is clarification_node_module.clarification_node
    )
