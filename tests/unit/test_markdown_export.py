from __future__ import annotations

import importlib
from pathlib import Path

from contentblitz.state import create_initial_state
from contentblitz.tools.exports.filenames import resolve_markdown_export_path
from contentblitz.tools.exports.markdown import build_markdown_export_document

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
                    "message": "{'code': 'configuration_error', 'message': 'OPENAI_API_KEY is not configured.', 'provider': 'openai', 'recoverable': False}",
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
        export_requested=True,
        export_metadata={
            "formats_requested": ["markdown"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    state.update(overrides)
    return state


def test_markdown_document_contains_expected_sections_and_deduped_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    markdown = build_markdown_export_document(_base_state(tmp_path))
    assert "# ContentBlitz Export" in markdown
    assert "## Workflow Summary" in markdown
    assert "## Blog Draft" in markdown
    assert "## LinkedIn Draft" in markdown
    assert "## Research Report" in markdown
    assert "## Image Prompts" in markdown
    assert "## Image Outputs" in markdown
    assert "## Sources" in markdown
    # source URL should only appear once after dedupe
    assert markdown.count("https://example.com/a") == 1
    assert "- `failed` | `dall-e-3` | Image generation encountered a recoverable issue." in markdown
    assert "{'code':" not in markdown
    assert "configuration_error" not in markdown
    assert "provider': 'openai'" not in markdown
    assert "recoverable': False" not in markdown
    assert "OPENAI_API_KEY" not in markdown


def test_markdown_workflow_summary_prefers_ui_workflow_status(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    markdown = build_markdown_export_document(
        _base_state(
            tmp_path,
            workflow_status="success",
            ui_workflow_status="partial_success",
        )
    )
    assert "- Workflow Status: `partial_success`" in markdown
    assert "- Workflow Status: `success`" not in markdown


def test_markdown_workflow_summary_uses_partial_success_for_degraded_nodes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    markdown = build_markdown_export_document(
        _base_state(
            tmp_path,
            workflow_status="success",
            ui_node_statuses={
                "research_agent_node": "degraded",
                "output_assembler_node": "completed",
                "export_node": "completed",
            },
            status_messages=[
                "Research results are degraded. Validate sources before publishing."
            ],
        )
    )
    assert "- Workflow Status: `partial_success`" in markdown
    assert "- Workflow Status: `success`" not in markdown
    assert "## Warnings" in markdown
    assert "Research results are degraded" in markdown


def test_markdown_workflow_summary_remains_success_when_clean(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    markdown = build_markdown_export_document(
        _base_state(
            tmp_path,
            workflow_status="success",
            image_outputs=[],
            research_data={"degraded": False},
            warnings=[],
            status_messages=[],
            errors=[],
            ui_node_statuses={
                "research_agent_node": "completed",
                "output_assembler_node": "completed",
                "export_node": "completed",
            },
        )
    )
    assert "- Workflow Status: `success`" in markdown


def test_markdown_export_node_creates_file_and_sets_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)

    metadata = updates["export_metadata"]
    markdown_path = metadata["export_paths"]["markdown"]
    assert markdown_path.endswith(".md")
    assert metadata["export_status"]["markdown"] == "completed"
    assert metadata["exported_at"]

    file_path = Path(markdown_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "## Sources" in content
    assert "Image generation encountered a recoverable issue." in content


def test_markdown_export_removes_sensitive_and_base64_payloads(tmp_path, monkeypatch) -> None:
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
    markdown_path = updates["export_metadata"]["export_paths"]["markdown"]
    file_path = Path(markdown_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    content = file_path.read_text(encoding="utf-8")

    assert "OPENAI_API_KEY" not in content
    assert "SERP_API_KEY" not in content
    assert "PERPLEXITY_API_KEY" not in content
    assert "Traceback" not in content
    assert "base64" not in content.lower()
    assert "data:image/" not in content.lower()


def test_markdown_export_downgrades_unsafe_links_but_preserves_safe_links(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(
        tmp_path,
        content_drafts={
            "blog": {
                "body": (
                    "Safe link [ref](https://example.com/ref) and "
                    "unsafe [x](javascript:alert(1)) and "
                    "![bad](data:image/png;base64,AAAA)"
                ),
                "version": 1,
            },
            "linkedin": {"body": "safe", "version": 1},
            "research_report": {"body": "safe"},
        },
    )
    updates = export_node_module.export_node(state)
    markdown_path = updates["export_metadata"]["export_paths"]["markdown"]
    file_path = Path(markdown_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    content = file_path.read_text(encoding="utf-8")

    assert "[ref](https://example.com/ref)" in content
    assert "javascript:" not in content.lower()
    assert "data:image/" not in content.lower()


def test_markdown_export_sanitizes_raw_provider_payloads_in_warnings(tmp_path, monkeypatch) -> None:
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
    markdown = build_markdown_export_document(state)
    lowered = markdown.lower()
    assert "configuration_error" not in lowered
    assert "provider': 'openai'" not in lowered
    assert "{'code':" not in lowered
    assert "a recoverable workflow issue was encountered." in lowered


def test_resolve_markdown_export_path_stays_inside_export_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    path = resolve_markdown_export_path("../unsafe/../../attempt")
    assert path.suffix == ".md"
    assert path.parent == (tmp_path / "exports").resolve()


def test_markdown_export_failure_is_non_blocking(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))

    def _fail_export(content: str, format_name: str):
        if format_name == "markdown":
            raise RuntimeError("OPENAI_API_KEY is missing")
        return {"path": f"exports/{format_name}"}

    monkeypatch.setattr(export_node_module, "export_content", _fail_export)
    state = _base_state(tmp_path)
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    assert metadata["export_status"]["markdown"] == "failed"
    assert metadata["export_paths"].get("markdown") is None
    assert metadata["error_log"]
    assert all("OPENAI_API_KEY" not in str(item) for item in metadata["error_log"])


def test_export_skipped_when_not_requested(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    state = _base_state(
        tmp_path,
        export_requested=False,
        export_metadata={
            "formats_requested": [],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    assert metadata["export_paths"] == {}
    assert metadata["export_status"] == {}
    assert metadata["error_log"] == []
