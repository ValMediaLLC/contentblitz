from __future__ import annotations

import importlib
from pathlib import Path

from contentblitz.state import create_initial_state
from contentblitz.tools.exports.filenames import resolve_html_export_path
from contentblitz.tools.exports.html import build_html_export_document

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
                    "recoverable": False,
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
            "formats_requested": ["html"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    state.update(overrides)
    return state


def test_html_document_contains_expected_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    html_doc = build_html_export_document(_base_state(tmp_path))
    lowered = html_doc.lower()
    assert "<!doctype html>" in lowered
    assert "<html" in lowered and "</html>" in lowered
    assert "<body" in lowered and "</body>" in lowered
    assert "workflow summary" in lowered
    assert "warnings" in lowered
    assert "blog draft" in lowered
    assert "linkedin draft" in lowered
    assert "research report" in lowered
    assert "image prompts" in lowered
    assert "image outputs" in lowered
    assert "sources" in lowered


def test_html_export_normalizes_image_errors_and_hides_provider_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    html_doc = build_html_export_document(_base_state(tmp_path))
    assert "Image generation failed safely." in html_doc
    assert "configuration_error" not in html_doc
    assert "OPENAI_API_KEY" not in html_doc
    assert "provider': 'openai'" not in html_doc
    assert "recoverable': False" not in html_doc
    assert "{'code':" not in html_doc


def test_html_export_sanitizes_scripts_event_handlers_and_js_urls(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(
        tmp_path,
        content_drafts={
            "blog": {
                "body": "<script>alert(1)</script><a href=\"javascript:alert(1)\" onclick=\"alert(1)\">click</a><iframe src=\"https://evil\"></iframe>",
                "version": 1,
            },
            "linkedin": {"body": "safe", "version": 1},
            "research_report": {"body": "safe"},
        },
    )
    html_doc = build_html_export_document(state)
    lowered = html_doc.lower()
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    assert "file://" not in lowered
    assert "data:text/html" not in lowered
    assert "onclick=" not in lowered
    assert "<iframe" not in lowered


def test_html_export_sanitizes_raw_provider_payloads_in_warnings(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(
        tmp_path,
        warnings=[],
        status_messages=[],
        errors=[
            {
                "message": "{'code': 'configuration_error', 'provider': 'openai', 'recoverable': False}",
                "recoverable": True,
            }
        ],
    )
    html_doc = build_html_export_document(state)
    lowered = html_doc.lower()
    assert "configuration_error" not in lowered
    assert "provider': 'openai'" not in lowered
    assert "{'code':" not in lowered
    assert "a recoverable workflow issue was encountered." in lowered


def test_html_export_node_creates_file_and_sets_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)

    metadata = updates["export_metadata"]
    html_path = metadata["export_paths"]["html"]
    assert html_path.endswith(".html")
    assert metadata["export_status"]["html"] == "completed"
    assert metadata["exported_at"]

    file_path = Path(html_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content.lower()


def test_html_export_failure_is_non_blocking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))

    def _fail_export(content: str, format_name: str):
        if format_name == "html":
            raise RuntimeError("OPENAI_API_KEY is missing")
        return {"path": f"exports/{format_name}"}

    monkeypatch.setattr(export_node_module, "export_content", _fail_export)
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    assert metadata["export_status"]["html"] == "failed"
    assert metadata["export_paths"].get("html") is None
    assert metadata["error_log"]
    assert all("OPENAI_API_KEY" not in str(item) for item in metadata["error_log"])


def test_resolve_html_export_path_stays_inside_export_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    path = resolve_html_export_path("../unsafe/../../attempt")
    assert path.suffix == ".html"
    assert path.parent == (tmp_path / "exports").resolve()
