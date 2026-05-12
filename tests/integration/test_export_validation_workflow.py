from __future__ import annotations

import importlib

from contentblitz.agents.output_assembler import output_assembler_node
from contentblitz.persistence.serialization import (
    deserialize_workflow_run,
    serialize_workflow_run,
)
from contentblitz.state import create_initial_state

export_node_module = importlib.import_module("contentblitz.agents.export_node")


def _workflow_state(*, export_requested: bool, formats_requested: list[str]) -> dict:
    return create_initial_state(
        user_query="create a blog article and linkedin post about AI marketing automation",
        requested_outputs=["blog", "linkedin", "research"],
        routing_decision="content_strategist_node",
        research_data={
            "summary": "Research summary for validation workflow checks.",
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
        sources=[
            {
                "title": "Source 1",
                "url": "https://example.com/source-1",
                "snippet": "Source 1 snippet",
                "citation_available": True,
                "credibility_score": 0.8,
            }
        ],
        status_messages=[],
        export_requested=export_requested,
        export_metadata={
            "formats_requested": formats_requested,
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )


def test_invalid_markdown_does_not_block_valid_html_export(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))

    def fake_markdown_validation(*args, **kwargs):
        return {"valid": False, "warnings": [], "errors": ["malformed markdown"]}

    monkeypatch.setattr(export_node_module, "validate_markdown_export", fake_markdown_validation)

    state = _workflow_state(export_requested=True, formats_requested=["markdown", "html"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}

    export_updates = export_node_module.export_node(merged)
    metadata = export_updates["export_metadata"]

    assert metadata["export_status"]["markdown"] == "failed"
    assert metadata["export_status"]["html"] == "completed"
    assert "markdown" not in metadata["export_paths"]
    assert "html" in metadata["export_paths"]
    assert metadata["export_error_count"] == 1


def test_validation_failure_metadata_restores_safely(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))

    def fake_markdown_validation(*args, **kwargs):
        return {"valid": False, "warnings": [], "errors": ["malformed markdown"]}

    monkeypatch.setattr(export_node_module, "validate_markdown_export", fake_markdown_validation)

    state = _workflow_state(export_requested=True, formats_requested=["markdown", "pdf"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}
    export_updates = export_node_module.export_node(merged)
    final_state = {**merged, **export_updates}

    persisted = serialize_workflow_run(
        result_state=final_state,
        ui_selected_options={
            "requested_outputs": ["blog", "linkedin", "research"],
            "export_requested": True,
            "export_formats": ["markdown", "pdf"],
        },
    )
    restored = deserialize_workflow_run(persisted)
    export_metadata = restored["export_metadata"]

    assert export_metadata["export_status"]["markdown"] == "failed"
    assert "markdown" not in export_metadata["export_paths"]
    assert export_metadata["export_status"]["pdf"] in {"completed", "failed"}


def test_invalid_citations_add_warning_without_blocking_safe_markdown_export(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))

    state = _workflow_state(export_requested=True, formats_requested=["markdown"])
    state["sources"] = [
        {
            "title": "Unsafe URL Source",
            "url": "javascript:alert(1)",
            "snippet": "Citation text is present but URL is unsafe.",
            "citation_available": True,
            "credibility_score": 0.4,
        }
    ]

    assembled = output_assembler_node(state)
    merged = {**state, **assembled}
    export_updates = export_node_module.export_node(merged)
    metadata = export_updates["export_metadata"]

    assert metadata["export_status"]["markdown"] == "completed"
    assert metadata["export_paths"]["markdown"].endswith(".md")
    warning_entries = [
        item
        for item in metadata.get("error_log", [])
        if item.get("code") == "markdown_validation_warning"
    ]
    assert warning_entries
    assert any("citation validation" in entry.get("message", "").lower() for entry in warning_entries)
