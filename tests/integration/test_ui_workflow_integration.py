from __future__ import annotations

from pathlib import Path

import pytest

from contentblitz.ui.rendering import build_render_payload
from contentblitz.ui.status import (
    apply_optional_node_skips,
    derive_node_statuses,
    summarize_workflow_status,
)
from tests.integration._ui_export_test_helpers import (
    apply_no_network,
    collect_stream_result,
    install_mock_image_client,
    install_mock_search,
    install_mock_text_client,
    reset_orchestrator_graph,
)


@pytest.fixture(autouse=True)
def _integration_guard(monkeypatch) -> None:
    apply_no_network(monkeypatch)
    reset_orchestrator_graph()


def _rendered_summary(result: dict, events: list[dict]) -> tuple[dict[str, str], dict]:
    node_statuses = apply_optional_node_skips(
        state=result,
        node_statuses=derive_node_statuses(events),
    )
    payload = build_render_payload(state=result, node_statuses=node_statuses)
    return node_statuses, payload


def test_blog_research_flow_renders_sources_usage_and_safe_output(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    install_mock_text_client(monkeypatch)
    install_mock_search(monkeypatch, weak_serp=False)

    events, result = collect_stream_result(
        user_query="create a blog article about AI productivity tools",
        requested_outputs=["blog", "research"],
        export_requested=False,
        export_formats=[],
    )

    assert result
    node_statuses, payload = _rendered_summary(result, events)
    summary = summarize_workflow_status(
        node_statuses,
        workflow_status=str(result.get("workflow_status", "")),
    )

    assert summary in {"success", "partial_success"}
    assert payload["partial_outputs"]["blog"].strip()
    assert payload["partial_outputs"]["research"].strip()
    assert payload["sources"]
    assert payload["usage_summary"]["search_queries"] > 0
    assert payload["usage_summary"]["estimated_tokens_out"] > 0
    assert payload["export_status"]["requested"] is False
    assert any(
        event.get("node_name") == "output_assembler_node"
        and event.get("status") == "completed"
        for event in events
    )

    combined_text = "\n".join(
        [
            payload.get("final_response", ""),
            *payload.get("warnings", []),
            *[
                item.get("message", "")
                for item in payload.get("errors", [])
                if isinstance(item, dict)
            ],
        ]
    ).lower()
    assert "traceback" not in combined_text
    assert "openai_api_key" not in combined_text
    assert "serp_api_key" not in combined_text
    assert "perplexity_api_key" not in combined_text


def test_multi_output_flow_keeps_image_failure_recoverable_and_exports_safe(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("SERP_API_KEY", "serp-test")
    monkeypatch.setenv("CONTENTBLITZ_EXPORT_DIR", str(tmp_path / "exports"))
    install_mock_text_client(monkeypatch)
    install_mock_search(monkeypatch, weak_serp=False)
    install_mock_image_client(monkeypatch, fail_all=True)

    events, result = collect_stream_result(
        user_query=(
            "create a detailed blog article, linkedin campaign, research report, and "
            "futuristic apparel image concepts about AI-native marketing agencies in 2030"
        ),
        requested_outputs=["blog", "linkedin", "research", "image"],
        export_requested=True,
        export_formats=["markdown", "html"],
    )

    assert result
    node_statuses, payload = _rendered_summary(result, events)
    summary = summarize_workflow_status(
        node_statuses,
        workflow_status=str(result.get("workflow_status", "")),
    )

    assert summary == "partial_success"
    assert payload["partial_outputs"]["blog"].strip()
    assert payload["partial_outputs"]["linkedin"].strip()
    assert payload["partial_outputs"]["research"].strip()
    assert payload["image_prompts"]
    assert any(item.get("status") == "failed" for item in payload["image_outputs"])
    assert any(
        (
            "recoverable" in warning.lower()
            or "image generation failed" in warning.lower()
        )
        for warning in payload["warnings"]
    )
    assert any(event.get("node_name") == "export_node" for event in events)

    export_status = payload["export_status"]
    assert export_status["requested"] is True
    assert export_status["paths"]
    for fmt, raw_path in export_status["paths"].items():
        safe_path = Path(raw_path)
        if not safe_path.is_absolute():
            safe_path = Path.cwd() / safe_path
        assert safe_path.exists()
        if fmt == "markdown":
            assert safe_path.suffix == ".md"
        if fmt == "html":
            assert safe_path.suffix == ".html"

    combined_text = "\n".join(
        [
            payload.get("final_response", ""),
            *payload.get("warnings", []),
            *[
                item.get("message", "")
                for item in payload.get("errors", [])
                if isinstance(item, dict)
            ],
        ]
    ).lower()
    assert "traceback" not in combined_text
    assert "<script" not in combined_text
    assert "openai_api_key" not in combined_text
