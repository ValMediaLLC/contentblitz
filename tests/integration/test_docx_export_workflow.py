from __future__ import annotations

import io
from pathlib import Path
import zipfile

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
            "degraded": True,
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
        status_messages=[
            "Research results are degraded. Validate sources before publishing."
        ],
        export_requested=export_requested,
        export_metadata={
            "formats_requested": formats_requested,
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )


def _read_docx_document_xml(docx_path: Path) -> str:
    with zipfile.ZipFile(io.BytesIO(docx_path.read_bytes()), mode="r") as archive:
        return archive.read("word/document.xml").decode("utf-8", errors="ignore")


def test_docx_export_pipeline_creates_file_and_preserves_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=True, formats_requested=["docx"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}

    export_updates = export_node(merged)
    final_state = {**merged, **export_updates}
    metadata = final_state["export_metadata"]

    assert metadata["export_status"]["docx"] == "completed"
    docx_path = metadata["export_paths"]["docx"]
    file_path = Path(docx_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    assert file_path.exists()

    document_xml = _read_docx_document_xml(file_path).lower()
    assert "workflow summary" in document_xml
    assert "warnings" in document_xml
    assert "blog draft" in document_xml
    assert "linkedin draft" in document_xml
    assert "research report" in document_xml
    assert "sources" in document_xml
    assert "openai_api_key" not in document_xml
    assert "traceback" not in document_xml
    assert "base64" not in document_xml


def test_docx_export_skipped_behavior_when_not_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=False, formats_requested=[])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}

    export_updates = export_node(merged)
    metadata = export_updates["export_metadata"]
    assert metadata["export_paths"] == {}
    assert metadata["export_status"] == {}


def test_persisted_session_restores_docx_export_metadata_safely(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=True, formats_requested=["docx"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}
    export_updates = export_node(merged)
    final_state = {**merged, **export_updates}

    persisted = serialize_workflow_run(
        result_state=final_state,
        ui_selected_options={
            "requested_outputs": ["blog", "linkedin", "research", "image"],
            "export_requested": True,
            "export_formats": ["docx"],
        },
    )
    restored = deserialize_workflow_run(persisted)

    export_metadata = restored["export_metadata"]
    assert export_metadata["formats_requested"] == ["docx"]
    assert "docx" in export_metadata["export_paths"]
    assert export_metadata["export_status"]["docx"] == "completed"

