from typing import Any

from contentblitz.state import create_initial_state
from contentblitz.agents.error_handler import error_handler_node


def _errors_are_nonfatal(result: dict[str, Any]) -> bool:
    errors = result.get("errors") or []
    return all(error.get("recoverable") is True for error in errors)


def test_error_handler_handles_fatal_error_state() -> None:
    state = create_initial_state(user_query="create a blog article about AI")
    state["errors"] = [
        {
            "agent": "blog_writer",
            "type": "unexpected_exception",
            "message": "Writer crashed.",
            "recoverable": False,
        }
    ]

    result = error_handler_node(state)

    assert result["workflow_status"] in {"failed", "error_handled"}
    assert result.get("final_response")
    assert "error" in result["final_response"].lower()


def test_error_handler_preserves_existing_errors() -> None:
    state = create_initial_state(user_query="create a blog article about AI")
    state["errors"] = [
        {
            "agent": "research_agent",
            "type": "provider_failure",
            "message": "Search provider unavailable.",
            "recoverable": False,
        }
    ]

    result = error_handler_node(state)

    assert result["errors"]
    assert result["errors"][0]["agent"] == "research_agent"
    assert result["errors"][0]["recoverable"] is False


def test_error_handler_handles_recoverable_errors_without_marking_fatal() -> None:
    state = create_initial_state(user_query="create some images")
    state["errors"] = [
        {
            "agent": "image_agent",
            "type": "image_generation_failed",
            "message": "No image assets returned.",
            "recoverable": True,
        }
    ]

    result = error_handler_node(state)

    assert result["workflow_status"] in {"error_handled", "completed_with_warnings"}
    assert result.get("final_response")
    assert _errors_are_nonfatal(result)


def test_error_handler_does_not_mutate_content_drafts() -> None:
    state = create_initial_state(user_query="create a blog article about AI")
    state["content_drafts"]["blog"] = {
        "body": "Existing draft",
        "version": 1,
    }
    state["errors"] = [
        {
            "agent": "quality_validator",
            "type": "validation_exception",
            "message": "Validation failed unexpectedly.",
            "recoverable": False,
        }
    ]

    result = error_handler_node(state)

    assert result["content_drafts"]["blog"]["body"] == "Existing draft"
    assert result["content_drafts"]["blog"]["version"] == 1


def test_error_handler_does_not_increment_retry_or_cost_counters() -> None:
    state = create_initial_state(user_query="create a blog article about AI")
    state["errors"] = [
        {
            "agent": "output_assembler",
            "type": "assembly_failed",
            "message": "Assembly failed.",
            "recoverable": False,
        }
    ]

    result = error_handler_node(state)

    assert result["cost_controls"]["total_retries_used_this_session"] == 0
    assert result["cost_controls"]["search_queries_used_this_session"] == 0
    assert result["cost_controls"]["image_generations_used_this_session"] == 0
    assert result["retry_counts"]["blog_writer"] == 0
    assert result["retry_counts"]["linkedin_writer"] == 0


def test_error_handler_handles_empty_error_list_safely() -> None:
    state = create_initial_state(user_query="create a blog article about AI")
    state["errors"] = []

    result = error_handler_node(state)

    assert result["workflow_status"] in {"error_handled", "completed_with_warnings"}
    assert result.get("final_response")