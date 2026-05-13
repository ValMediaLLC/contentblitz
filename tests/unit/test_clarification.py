from __future__ import annotations

import importlib

from contentblitz.state import create_initial_state

clarification_module = importlib.import_module("contentblitz.agents.clarification")


def _base_state(**overrides):
    state = create_initial_state(user_query="help me")
    state.update(overrides)
    return state


def test_extract_message_appends_question_mark_for_plain_text() -> None:
    message = clarification_module._extract_message({"output": "Could you provide more detail"})
    assert message == "Could you provide more detail?"


def test_extract_message_from_nested_mapping_prefers_known_fields() -> None:
    response = {"output": {"message": "Which audience should this target"}}
    message = clarification_module._extract_message(response)
    assert message == "Which audience should this target?"


def test_extract_message_returns_empty_for_unusable_shape() -> None:
    message = clarification_module._extract_message({"output": {"x": 123}})
    assert message == ""


def test_resolve_clarification_message_preserves_existing_value(monkeypatch) -> None:
    state = _base_state(clarification_message="What format should I generate?")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("generate_text should not be called when message already exists.")

    monkeypatch.setattr(clarification_module, "generate_text", _fail_if_called)
    assert (
        clarification_module._resolve_clarification_message(state)
        == "What format should I generate?"
    )


def test_resolve_clarification_message_uses_static_fallback_on_tool_failure(
    monkeypatch,
) -> None:
    state = _base_state(user_query="   ", clarification_message=None)

    def _raise_error(*_args, **_kwargs):
        raise RuntimeError("tool failure")

    monkeypatch.setattr(clarification_module, "generate_text", _raise_error)
    message = clarification_module._resolve_clarification_message(state)
    assert (
        message
        == "Could you clarify your goal, target audience, and desired output format?"
    )


def test_clarification_node_handles_empty_prompt_safely_without_draft_mutation() -> None:
    state = _base_state(user_query="   ", clarification_message="Can you clarify your request?")
    updates = clarification_module.clarification_node(state)

    assert updates["workflow_status"] == "awaiting_clarification"
    assert updates["final_response"] == "Can you clarify your request?"
    assert updates["conversation_history"][-1] == {"role": "user", "content": ""}
    assert "content_drafts" not in updates

