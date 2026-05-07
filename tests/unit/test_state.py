from contentblitz.config import RETRY_POLICY
from contentblitz.state import ContentBlitzState, create_initial_state


def test_state_contains_all_top_level_fields_from_spec() -> None:
    state = create_initial_state()
    expected_keys = {
        "session_id",
        "user_id",
        "user_query",
        "intent",
        "routing_decision",
        "requested_outputs",
        "conversation_history",
        "research_required",
        "clarification_needed",
        "clarification_message",
        "research_data",
        "sources",
        "content_brief",
        "content_drafts",
        "draft_status",
        "best_drafts",
        "attempt_history",
        "retry_feedback",
        "retry_counts",
        "quality_scores",
        "image_prompts",
        "image_outputs",
        "tool_outputs",
        "errors",
        "final_response",
        "workflow_status",
        "export_requested",
        "export_metadata",
        "cache_metadata",
        "cost_controls",
    }
    assert set(state.keys()) == expected_keys


def test_state_nested_shapes_match_spec_defaults() -> None:
    state = create_initial_state()

    assert set(state["content_brief"].keys()) == {"blog", "linkedin", "image"}
    assert state["content_drafts"]["blog"] == {"body": "", "version": 0}
    assert state["content_drafts"]["linkedin"] == {"body": "", "version": 0}
    assert state["content_drafts"]["research_report"] == {"body": ""}
    assert state["draft_status"] == {}
    assert state["best_drafts"] == {"blog": None, "linkedin": None}
    assert set(state["attempt_history"].keys()) == {"blog", "linkedin", "image"}
    assert set(state["retry_feedback"].keys()) == {"blog", "linkedin"}

    assert set(state["retry_counts"].keys()) == set(RETRY_POLICY.keys())
    assert all(value == 0 for value in state["retry_counts"].values())

    assert state["export_metadata"] == {
        "formats_requested": [],
        "export_paths": {},
        "exported_at": None,
        "error_log": [],
    }
    assert state["cache_metadata"] == {
        "enabled": True,
        "ttl_seconds": 1800,
        "backend": "in_memory",
        "keys": [],
    }
    assert state["cost_controls"] == {
        "tokens_used_this_session": 0,
        "search_queries_used_this_session": 0,
        "image_generations_used_this_session": 0,
        "total_retries_used_this_session": 0,
        "budget_exceeded": False,
    }


def test_state_dataclass_is_instantiable() -> None:
    state = ContentBlitzState()
    assert state.session_id == ""
