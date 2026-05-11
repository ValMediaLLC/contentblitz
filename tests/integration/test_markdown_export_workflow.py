from __future__ import annotations

from pathlib import Path

from contentblitz.agents.export_node import export_node
from contentblitz.agents.output_assembler import output_assembler_node
from contentblitz.persistence.serialization import (
    deserialize_workflow_run,
    serialize_workflow_run,
)
from contentblitz.state import create_initial_state


def _workflow_state(*, export_requested: bool, formats_requested: list[str]) -> dict:
    return create_initial_state(
        user_query="create a blog article, linkedin post, and image concept about AI marketing automation",
        requested_outputs=["blog", "linkedin", "research", "image"],
        routing_decision="content_strategist_node",
        research_data={
            "summary": "Research summary for integration export validation.",
            "degraded": False,
        },
        content_drafts={
            "blog": {"body": "Blog integration draft body", "version": 1},
            "linkedin": {"body": "LinkedIn integration draft body", "version": 1},
            "research_report": {"body": "Research report integration body"},
        },
        best_drafts={
            "blog": {"body": "Blog integration draft body", "composite": 0.91, "version": 1},
            "linkedin": {"body": "LinkedIn integration draft body", "composite": 0.9, "version": 1},
        },
        quality_scores={
            "blog": {"validation_status": "passed", "composite": 0.91},
            "linkedin": {"validation_status": "passed", "composite": 0.9},
        },
        image_prompts=["Create cinematic AI marketing concept art."],
        image_outputs=[
            {
                "status": "failed",
                "provider": "dall-e-3",
                "error": {"message": "OPENAI_API_KEY is not configured", "recoverable": True},
            }
        ],
        sources=[
            {
                "title": "Source 1",
                "url": "https://example.com/source-1",
                "snippet": "Source 1 snippet",
                "citation_available": True,
                "credibility_score": 0.8,
            }
        ],
        warnings=["None"],
        export_requested=export_requested,
        export_metadata={
            "formats_requested": formats_requested,
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )


def test_markdown_export_pipeline_creates_file_and_preserves_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=True, formats_requested=["markdown"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}

    export_updates = export_node(merged)
    final_state = {**merged, **export_updates}
    metadata = final_state["export_metadata"]

    assert metadata["export_status"]["markdown"] == "completed"
    markdown_path = metadata["export_paths"]["markdown"]
    file_path = Path(markdown_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    assert file_path.exists()

    content = file_path.read_text(encoding="utf-8")
    assert "# ContentBlitz Export" in content
    assert "## Workflow Summary" in content
    assert "## Blog Draft" in content
    assert "## LinkedIn Draft" in content
    assert "## Research Report" in content
    assert "## Sources" in content
    assert "OPENAI_API_KEY" not in content
    assert "Traceback" not in content
    assert "base64" not in content.lower()
    assert "{'code':" not in content
    assert "configuration_error" not in content
    assert "provider': 'openai'" not in content
    assert "recoverable': False" not in content
    assert "- Workflow Status: `partial_success`" in content


def test_export_skipped_behavior_no_markdown_path_when_not_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=False, formats_requested=[])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}

    export_updates = export_node(merged)
    metadata = export_updates["export_metadata"]
    assert metadata["export_paths"] == {}
    assert metadata["export_status"] == {}


def test_markdown_workflow_summary_uses_aggregated_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=True, formats_requested=["markdown"])
    state["workflow_status"] = "success"
    state["ui_workflow_status"] = "partial_success"
    state["ui_node_statuses"] = {
        "research_agent_node": "degraded",
        "output_assembler_node": "completed",
        "export_node": "completed",
    }
    state["research_data"]["degraded"] = True
    state["status_messages"] = [
        "Research results are degraded. Validate sources before publishing."
    ]

    export_updates = export_node(state)
    metadata = export_updates["export_metadata"]
    markdown_path = metadata["export_paths"]["markdown"]
    file_path = Path(markdown_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    content = file_path.read_text(encoding="utf-8")

    assert "- Workflow Status: `partial_success`" in content
    assert "- Workflow Status: `success`" not in content
    assert "## Warnings" in content
    assert "Research results are degraded" in content


def test_persisted_session_restores_export_metadata_safely(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=True, formats_requested=["markdown"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}
    export_updates = export_node(merged)
    final_state = {**merged, **export_updates}

    persisted = serialize_workflow_run(
        result_state=final_state,
        ui_selected_options={
            "requested_outputs": ["blog", "linkedin", "research", "image"],
            "export_requested": True,
            "export_formats": ["markdown"],
        },
    )
    restored = deserialize_workflow_run(persisted)

    export_metadata = restored["export_metadata"]
    assert export_metadata["formats_requested"] == ["markdown"]
    assert "markdown" in export_metadata["export_paths"]
    assert export_metadata["export_status"]["markdown"] == "completed"
