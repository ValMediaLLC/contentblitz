from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from contentblitz.ui.rendering import build_render_payload
from frontend import session as session_module
from frontend.pages import history as history_page


@dataclass
class _DummyStreamlit:
    session_state: Dict[str, Any] = field(default_factory=dict)


def _install_dummy_streamlit(monkeypatch) -> _DummyStreamlit:
    dummy = _DummyStreamlit()
    monkeypatch.setattr(session_module, "st", dummy)
    return dummy


def _sample_result(*, workflow_status: str, user_query: str) -> dict:
    return {
        "user_query": user_query,
        "requested_outputs": ["blog"],
        "workflow_status": workflow_status,
        "ui_workflow_status": workflow_status,
        "routing_decision": "content_strategist_node",
        "final_response": "Saved final response.",
        "content_drafts": {
            "blog": {"body": "Blog body", "version": 1},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        "partial_outputs": {"blog": "Blog body", "linkedin": "", "research": ""},
        "partial_output_mode": "blog_only",
        "image_prompts": [],
        "image_outputs": [],
        "sources": [],
        "quality_scores": {},
        "export_metadata": {"formats_requested": [], "export_paths": {}},
        "errors": [],
        "warnings": [],
    }


def test_saves_lists_and_restores_partial_success_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_SESSION_DIR", str(tmp_path / "sessions"))
    _install_dummy_streamlit(monkeypatch)
    session_module.initialize_session_state()

    result = _sample_result(
        workflow_status="partial_success",
        user_query="Write a short blog post on AI workflow orchestration.",
    )
    run_id = session_module.save_persisted_run(
        result=result,
        last_submission={
            "requested_outputs": ["blog"],
            "export_requested": False,
            "export_formats": [],
        },
        progress_events=[
            {
                "node_name": "query_handler_node",
                "status": "completed",
                "message": "done",
                "timestamp": "2026-05-10T10:00:00+00:00",
                "safe_metadata": {},
            }
        ],
        node_statuses={"query_handler_node": "completed", "blog_writer_node": "completed"},
        status_messages=["Workflow completed with recoverable warnings."],
    )
    assert run_id

    summaries = session_module.list_persisted_run_summaries()
    assert len(summaries) == 1
    assert summaries[0]["workflow_status"] == "partial_success"
    assert "partial_success" in history_page._summary_label(summaries[0])

    restored, _ = session_module.restore_persisted_run(run_id)
    assert restored is True
    assert session_module.get_execution_status() == "partial_success"
    restored_result = session_module.get_last_result()
    assert isinstance(restored_result, dict)
    assert restored_result["run_id"] == run_id
    payload = build_render_payload(
        state=restored_result,
        node_statuses=session_module.get_node_statuses(),
    )
    assert payload["final_response"] == "Saved final response."


def test_restores_awaiting_clarification_without_reexecution(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_SESSION_DIR", str(tmp_path / "sessions"))
    _install_dummy_streamlit(monkeypatch)
    session_module.initialize_session_state()

    result = _sample_result(
        workflow_status="awaiting_clarification",
        user_query="help",
    )
    result["final_response"] = "Could you clarify your goal, target audience, and desired output format?"

    run_id = session_module.save_persisted_run(
        result=result,
        last_submission={
            "requested_outputs": ["blog"],
            "export_requested": False,
            "export_formats": [],
        },
        progress_events=[],
        node_statuses={
            "query_handler_node": "completed",
            "clarification_node": "completed",
        },
        status_messages=["Workflow paused awaiting clarification."],
    )
    assert run_id

    restored, _ = session_module.restore_persisted_run(run_id)
    assert restored is True
    assert session_module.get_execution_status() == "awaiting_clarification"
    assert any(
        "awaiting clarification" in message.lower()
        for message in session_module.get_status_messages()
    )
