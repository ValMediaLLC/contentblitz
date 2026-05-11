from __future__ import annotations

import importlib
import io
from pathlib import Path
import zipfile

from contentblitz.state import create_initial_state
from contentblitz.tools.exports.docx import build_docx_export_document
from contentblitz.tools.exports.filenames import resolve_docx_export_path
from contentblitz.tools.exports.validation import validate_docx_export

export_node_module = importlib.import_module("contentblitz.agents.export_node")


def _read_docx_document_xml(payload: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
        return archive.read("word/document.xml").decode("utf-8", errors="ignore")


def _base_state(tmp_path: Path, **overrides):
    state = create_initial_state(
        user_query="Create content.",
        requested_outputs=["blog", "linkedin", "research", "image"],
        routing_decision="content_strategist_node",
        workflow_status="partial_success",
        final_response="Final response body.",
        assembled_outputs={
            "blog": "Blog assembled output.",
            "linkedin": "LinkedIn assembled output.",
            "research": "Research assembled output.",
            "image": "Image assembled output.",
        },
        content_drafts={
            "blog": {"body": "Blog draft body", "version": 1},
            "linkedin": {"body": "LinkedIn draft body", "version": 1},
            "research_report": {"body": "Research report body"},
        },
        image_prompts=["Generate a modern concept image."],
        image_outputs=[
            {
                "status": "failed",
                "provider": "dall-e-3",
                "error": {
                    "code": "configuration_error",
                    "message": "{'code': 'configuration_error', 'message': 'OPENAI_API_KEY is not configured.', 'provider': 'openai', 'recoverable': False}",
                    "provider": "openai",
                    "recoverable": True,
                },
            }
        ],
        sources=[
            {
                "title": "Source A",
                "url": "https://example.com/a",
                "snippet": "A source snippet",
                "citation_available": True,
                "credibility_score": 0.9,
            },
            {
                "title": "Source A duplicate",
                "url": "https://example.com/a",
                "snippet": "Duplicate source snippet",
                "citation_available": True,
                "credibility_score": 0.4,
            },
        ],
        warnings=["Research results are degraded and may require manual verification."],
        status_messages=["Workflow completed with recoverable warnings."],
        export_requested=True,
        export_metadata={
            "formats_requested": ["docx"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    state.update(overrides)
    return state


def test_docx_document_contains_expected_sections_and_markers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    docx_bytes = build_docx_export_document(_base_state(tmp_path))
    assert docx_bytes.startswith(b"PK")
    result = validate_docx_export(docx_bytes, sources_exist=True)
    assert result["valid"] is True

    text = _read_docx_document_xml(docx_bytes).lower()
    assert "workflow summary" in text
    assert "warnings" in text
    assert "blog draft" in text
    assert "linkedin draft" in text
    assert "research report" in text
    assert "image prompts" in text
    assert "image outputs" in text
    assert "sources" in text
    assert "configuration_error" not in text
    assert "openai_api_key" not in text
    assert "{'code':" not in text


def test_docx_export_node_creates_file_and_sets_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)

    metadata = updates["export_metadata"]
    docx_path = metadata["export_paths"]["docx"]
    assert docx_path.endswith(".docx")
    assert metadata["export_status"]["docx"] == "completed"
    assert metadata["exported_at"]

    file_path = Path(docx_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    assert file_path.exists()
    content = file_path.read_bytes()
    assert content.startswith(b"PK")


def test_docx_export_removes_sensitive_and_base64_payloads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(
        tmp_path,
        image_outputs=[
            {
                "status": "failed",
                "provider": "dall-e-3",
                "url": "data:image/png;base64,AAAA",
                "error": {
                    "message": "Traceback (most recent call last): OPENAI_API_KEY=sk-secret",
                    "recoverable": True,
                },
            }
        ],
        warnings=["SERP_API_KEY=serp-secret"],
        status_messages=["PERPLEXITY_API_KEY=pplx-secret"],
    )
    updates = export_node_module.export_node(state)
    docx_path = updates["export_metadata"]["export_paths"]["docx"]
    file_path = Path(docx_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    content = _read_docx_document_xml(file_path.read_bytes())
    lowered = content.lower()

    assert "OPENAI_API_KEY" not in content
    assert "SERP_API_KEY" not in content
    assert "PERPLEXITY_API_KEY" not in content
    assert "Traceback" not in content
    assert "base64" not in lowered
    assert "data:image/" not in lowered


def test_docx_export_failure_is_non_blocking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))

    def _fail_export(content: str, format_name: str):
        if format_name == "docx":
            raise RuntimeError("OPENAI_API_KEY is missing")
        return {"path": f"exports/{format_name}"}

    monkeypatch.setattr(export_node_module, "export_content", _fail_export)
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    assert metadata["export_status"]["docx"] == "failed"
    assert metadata["export_paths"].get("docx") is None
    assert metadata["error_log"]
    assert all("OPENAI_API_KEY" not in str(item) for item in metadata["error_log"])


def test_validate_docx_export_rejects_sensitive_payloads() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                "<Override PartName=\"/word/document.xml\" "
                "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                "<Relationship Id=\"rId1\" "
                "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
                "Target=\"word/document.xml\"/>"
                "</Relationships>"
            ),
        )
        archive.writestr(
            "word/document.xml",
            (
                "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
                "<w:body><w:p><w:r><w:t>Traceback (most recent call last): OPENAI_API_KEY=sk-secret data:image/png;base64,AAAA</w:t></w:r></w:p></w:body>"
                "</w:document>"
            ),
        )

    result = validate_docx_export(buffer.getvalue(), sources_exist=False)
    assert result["valid"] is False
    joined = " ".join(result["errors"]).lower()
    assert "stack trace" in joined
    assert "environment variable" in joined
    assert "base64" in joined


def test_resolve_docx_export_path_stays_inside_export_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    path = resolve_docx_export_path("../unsafe/../../attempt")
    assert path.suffix == ".docx"
    assert path.parent == (tmp_path / "exports").resolve()

