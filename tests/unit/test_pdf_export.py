from __future__ import annotations

import importlib
from pathlib import Path

from contentblitz.state import create_initial_state
from contentblitz.tools.exports.filenames import resolve_pdf_export_path
from contentblitz.tools.exports.pdf import build_pdf_export_document

export_node_module = importlib.import_module("contentblitz.agents.export_node")


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
            "formats_requested": ["pdf"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    state.update(overrides)
    return state


def test_pdf_document_contains_expected_sections_and_markers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    pdf_bytes = build_pdf_export_document(_base_state(tmp_path))
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"xref" in pdf_bytes
    assert b"trailer" in pdf_bytes
    assert b"%%EOF" in pdf_bytes

    text = pdf_bytes.decode("latin-1", errors="ignore")
    lowered = text.lower()
    assert "workflow summary" in lowered
    assert "warnings" in lowered
    assert "blog draft" in lowered
    assert "linkedin draft" in lowered
    assert "research report" in lowered
    assert "image prompts" in lowered
    assert "image outputs" in lowered
    assert "sources" in lowered
    assert "configuration_error" not in lowered
    assert "openai_api_key" not in lowered
    assert "{'code':" not in lowered


def test_pdf_export_node_creates_file_and_sets_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)

    metadata = updates["export_metadata"]
    pdf_path = metadata["export_paths"]["pdf"]
    assert pdf_path.endswith(".pdf")
    assert metadata["export_status"]["pdf"] == "completed"
    assert metadata["exported_at"]

    file_path = Path(pdf_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    assert file_path.exists()
    content = file_path.read_bytes()
    assert content.startswith(b"%PDF-")
    assert content.rstrip().endswith(b"%%EOF")


def test_pdf_export_removes_sensitive_and_base64_payloads(
    tmp_path, monkeypatch
) -> None:
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
    pdf_path = updates["export_metadata"]["export_paths"]["pdf"]
    file_path = Path(pdf_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    content = file_path.read_bytes().decode("latin-1", errors="ignore")

    assert "OPENAI_API_KEY" not in content
    assert "SERP_API_KEY" not in content
    assert "PERPLEXITY_API_KEY" not in content
    assert "Traceback" not in content
    assert "base64" not in content.lower()
    assert "data:image/" not in content.lower()


def test_pdf_export_failure_is_non_blocking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))

    def _fail_export(content: str, format_name: str):
        if format_name == "pdf":
            raise RuntimeError("OPENAI_API_KEY is missing")
        return {"path": f"exports/{format_name}"}

    monkeypatch.setattr(export_node_module, "export_content", _fail_export)
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    assert metadata["export_status"]["pdf"] == "failed"
    assert metadata["export_paths"].get("pdf") is None
    assert metadata["error_log"]
    assert all("OPENAI_API_KEY" not in str(item) for item in metadata["error_log"])


def test_resolve_pdf_export_path_stays_inside_export_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    path = resolve_pdf_export_path("../unsafe/../../attempt")
    assert path.suffix == ".pdf"
    assert path.parent == (tmp_path / "exports").resolve()
