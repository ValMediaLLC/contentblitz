from contentblitz.agents.error_handler import error_handler_node
from contentblitz.state import create_initial_state


def test_empty_errors_creates_generic_failure_message() -> None:
    state = create_initial_state(errors=[])
    result = error_handler_node(state)

    assert result["final_response"]
    assert "unexpected error" in result["final_response"].lower()
    assert result["workflow_status"] in {"error_handled", "failed"}


def test_unrecoverable_errors_create_safe_failure_message() -> None:
    state = create_initial_state(
        errors=[
            {
                "agent": "research_agent",
                "type": "provider_failure",
                "message": "provider stack trace details here",
                "recoverable": False,
            }
        ]
    )
    result = error_handler_node(state)

    assert result["workflow_status"] == "failed"
    assert "unexpected error" in result["final_response"].lower()
    assert "stack trace" not in result["final_response"].lower()


def test_api_key_like_strings_are_not_exposed() -> None:
    state = create_initial_state(
        errors=[
            {
                "agent": "query_handler",
                "type": "unexpected_exception",
                "message": "OPENAI_API_KEY=sk-test-1234 secret leaked",
                "recoverable": False,
            }
        ]
    )
    result = error_handler_node(state)
    response = result["final_response"]

    assert "sk-test-1234" not in response
    assert "openai_api_key" not in response.lower()
    assert "secret leaked" not in response.lower()


def test_terminal_error_entry_appended() -> None:
    state = create_initial_state(
        errors=[
            {
                "agent": "blog_writer",
                "type": "unexpected_exception",
                "message": "Writer crashed",
                "recoverable": False,
            }
        ]
    )
    result = error_handler_node(state)

    assert len(result["errors"]) == 2
    terminal = result["errors"][-1]
    assert terminal["agent"] == "error_handler"
    assert terminal["type"] == "terminal_error"
    assert terminal["recoverable"] is False


def test_workflow_status_is_failed_for_fatal_errors() -> None:
    state = create_initial_state(
        errors=[
            {
                "agent": "output_assembler",
                "type": "assembly_failed",
                "message": "internal failure",
                "recoverable": False,
            }
        ]
    )
    result = error_handler_node(state)

    assert result["workflow_status"] == "failed"
