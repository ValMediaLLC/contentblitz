from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from contentblitz.core import observability as observability_module
from contentblitz.ui.rendering import build_render_payload
from contentblitz.ui.status import apply_optional_node_skips, derive_node_statuses
from tests.integration._ui_export_test_helpers import (
    apply_no_network,
    collect_stream_result,
    install_mock_image_client,
    install_mock_search,
    install_mock_text_client,
    reset_orchestrator_graph,
)

export_node_module = importlib.import_module("contentblitz.agents.export_node")


@pytest.fixture(autouse=True)
def _integration_guard(monkeypatch) -> None:
    apply_no_network(monkeypatch)
    reset_orchestrator_graph()


def _render_payload(result: dict, events: list[dict]) -> dict:
    node_statuses = apply_optional_node_skips(
        state=result,
        node_statuses=derive_node_statuses(events),
    )
    return build_render_payload(state=result, node_statuses=node_statuses)


def test_export_pipeline_integrates_markdown_html_pdf_docx(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    install_mock_text_client(monkeypatch)
    install_mock_search(monkeypatch, weak_serp=False)
    install_mock_image_client(monkeypatch, fail_all=False)

    events, result = collect_stream_result(
        user_query=(
            "create a detailed blog article, linkedin campaign, research report, and "
            "futuristic apparel image concepts about AI-native marketing agencies "
            "in 2030"
        ),
        requested_outputs=["blog", "linkedin", "research", "image"],
        export_requested=True,
        export_formats=["markdown", "html", "pdf", "docx"],
    )

    assert result
    metadata = result.get("export_metadata", {})
    assert isinstance(metadata, dict)
    assert set(metadata.get("formats_requested", [])) >= {
        "markdown",
        "html",
        "pdf",
        "docx",
    }

    export_status = metadata.get("export_status", {})
    export_paths = metadata.get("export_paths", {})
    assert isinstance(export_status, dict)
    assert isinstance(export_paths, dict)

    for fmt in ("markdown", "html", "pdf", "docx"):
        assert export_status.get(fmt) == "completed"
        path_value = export_paths.get(fmt, "")
        assert isinstance(path_value, str) and path_value.strip()
        safe_path = Path(path_value)
        if not safe_path.is_absolute():
            safe_path = Path.cwd() / safe_path
        assert safe_path.exists()
        if fmt == "markdown":
            assert safe_path.suffix == ".md"
            assert "openai_api_key" not in safe_path.read_text(encoding="utf-8").lower()
        if fmt == "html":
            assert safe_path.suffix == ".html"
            assert "openai_api_key" not in safe_path.read_text(encoding="utf-8").lower()
        if fmt == "pdf":
            assert safe_path.suffix == ".pdf"
            assert safe_path.read_bytes().startswith(b"%PDF-")
        if fmt == "docx":
            assert safe_path.suffix == ".docx"

    payload = _render_payload(result, events)
    assert payload["export_status"]["requested"] is True
    assert set(payload["export_status"]["paths"]).issubset(
        {"markdown", "html", "pdf", "docx"}
    )
    assert "traceback" not in payload["final_response"].lower()


def test_export_validation_failure_marks_only_failing_format(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    install_mock_text_client(monkeypatch)
    install_mock_search(monkeypatch, weak_serp=False)
    install_mock_image_client(monkeypatch, fail_all=False)

    monkeypatch.setattr(
        export_node_module,
        "validate_markdown_export",
        lambda *args, **kwargs: {
            "valid": False,
            "warnings": [],
            "errors": ["malformed markdown"],
        },
    )

    _, result = collect_stream_result(
        user_query="create a blog article about AI productivity tools",
        requested_outputs=["blog", "research"],
        export_requested=True,
        export_formats=["markdown", "html"],
    )

    metadata = result.get("export_metadata", {})
    assert isinstance(metadata, dict)
    export_status = metadata.get("export_status", {})
    export_paths = metadata.get("export_paths", {})
    assert export_status.get("markdown") == "failed"
    assert export_status.get("html") == "completed"
    assert "markdown" not in export_paths
    assert "html" in export_paths

    error_log = metadata.get("error_log", [])
    assert isinstance(error_log, list) and error_log
    flattened = "\n".join(
        str(item.get("message", "")) for item in error_log if isinstance(item, dict)
    ).lower()
    assert "traceback" not in flattened
    assert "openai_api_key" not in flattened


def test_pdf_export_with_image_success_keeps_consistent_success_state(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    install_mock_text_client(monkeypatch)
    install_mock_search(monkeypatch, weak_serp=False)
    install_mock_image_client(monkeypatch, fail_all=False)

    events, result = collect_stream_result(
        user_query=(
            "Create a long research-backed blog post, LinkedIn post, and image "
            "about the best electric cars to buy in 2026, include sources, then "
            "export everything as a PDF with the image embedded."
        ),
        requested_outputs=["blog", "linkedin", "research", "image"],
        export_requested=True,
        export_formats=["pdf"],
    )

    metadata = result.get("export_metadata", {})
    assert isinstance(metadata, dict)
    assert metadata.get("export_status", {}).get("pdf") == "completed"
    assert metadata.get("export_error_count") == 0
    assert metadata.get("failed_export_formats", []) == []
    assert metadata.get("completed_export_formats", []) == ["pdf"]
    export_terminal_events = [
        event
        for event in events
        if str(event.get("node_name", "")).strip() == "export_node"
        and str(event.get("status", "")).strip() in {"completed", "degraded", "failed"}
    ]
    assert export_terminal_events
    latest_export_event = export_terminal_events[-1]
    assert latest_export_event["status"] == "completed"
    assert (
        latest_export_event.get("safe_metadata", {}).get("export_error_count", 0) == 0
    )

    node_statuses = apply_optional_node_skips(
        state=result,
        node_statuses=derive_node_statuses(events),
    )
    assert node_statuses.get("export_node") == "completed"
    assert str(result.get("workflow_status", "")).strip().lower() == "success"

    payload = _render_payload(result, events)
    assert payload["export_status"]["export_error_count"] == 0
    assert payload["export_status"]["non_blocking_failure"] is False

    trace_metadata = observability_module.safe_trace_metadata(result)
    assert trace_metadata["workflow_status"] == "success"
    assert trace_metadata["export_failure_status"] is False
