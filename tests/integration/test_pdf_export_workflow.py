from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path

from contentblitz.agents.export_node import export_node
from contentblitz.agents.output_assembler import output_assembler_node
from contentblitz.persistence.serialization import (
    deserialize_workflow_run,
    serialize_workflow_run,
)
from contentblitz.state import create_initial_state


def _write_test_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1
    height = 1
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\xff")  # blue pixel

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def _workflow_state(*, export_requested: bool, formats_requested: list[str]) -> dict:
    return create_initial_state(
        user_query=(
            "create a blog article, linkedin post, and image concept "
            "about AI marketing automation"
        ),
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
            "blog": {
                "body": "Blog integration draft body",
                "composite": 0.91,
                "version": 1,
            },
            "linkedin": {
                "body": "LinkedIn integration draft body",
                "composite": 0.9,
                "version": 1,
            },
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
                "error": {
                    "message": "OPENAI_API_KEY is not configured",
                    "recoverable": True,
                },
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


def test_pdf_export_pipeline_creates_file_and_preserves_sections(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=True, formats_requested=["pdf"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}

    export_updates = export_node(merged)
    final_state = {**merged, **export_updates}
    metadata = final_state["export_metadata"]

    assert metadata["export_status"]["pdf"] == "completed"
    pdf_path = metadata["export_paths"]["pdf"]
    file_path = Path(pdf_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    assert file_path.exists()

    content = file_path.read_bytes()
    text = content.decode("latin-1", errors="ignore").lower()
    assert content.startswith(b"%PDF-")
    assert b"%%EOF" in content
    assert "workflow summary" in text
    assert "warnings" in text
    assert "blog draft" in text
    assert "linkedin draft" in text
    assert "research report" in text
    assert "sources" in text
    assert "openai_api_key" not in text
    assert "traceback" not in text
    assert "base64" not in text


def test_pdf_export_skipped_behavior_when_not_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=False, formats_requested=[])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}

    export_updates = export_node(merged)
    metadata = export_updates["export_metadata"]
    assert metadata["export_paths"] == {}
    assert metadata["export_status"] == {}


def test_image_only_pdf_export_uses_local_path_and_status_not_failed(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    local_image = tmp_path / "exports" / "images" / "lookbook.png"
    _write_test_png(local_image)
    state = create_initial_state(
        user_query="Create futuristic fashion image concepts.",
        requested_outputs=["image"],
        routing_decision="image_agent_node",
        content_drafts={
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        image_prompts=["Create a neon fashion lookbook image."],
        image_outputs=[
            {
                "status": "success",
                "provider": "gpt-image-1",
                "local_path": str(local_image),
                "renderable": True,
            }
        ],
        sources=[],
        export_requested=True,
        export_metadata={
            "formats_requested": ["pdf"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}
    assert merged["workflow_status"] != "failed"

    export_updates = export_node(merged)
    metadata = export_updates["export_metadata"]
    assert metadata["export_status"]["pdf"] == "completed"
    pdf_path = metadata["export_paths"]["pdf"]
    file_path = Path(pdf_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    text = file_path.read_bytes().decode("latin-1", errors="ignore").lower()
    assert "image outputs" in text
    assert "lookbook.png" in text
    assert "/subtype /image" in text


def test_persisted_session_restores_pdf_export_metadata_safely(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _workflow_state(export_requested=True, formats_requested=["pdf"])
    assembled = output_assembler_node(state)
    merged = {**state, **assembled}
    export_updates = export_node(merged)
    final_state = {**merged, **export_updates}

    persisted = serialize_workflow_run(
        result_state=final_state,
        ui_selected_options={
            "requested_outputs": ["blog", "linkedin", "research", "image"],
            "export_requested": True,
            "export_formats": ["pdf"],
        },
    )
    restored = deserialize_workflow_run(persisted)

    export_metadata = restored["export_metadata"]
    assert export_metadata["formats_requested"] == ["pdf"]
    assert "pdf" in export_metadata["export_paths"]
    assert export_metadata["export_status"]["pdf"] == "completed"
