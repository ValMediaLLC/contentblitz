import importlib

export_node_module = importlib.import_module("contentblitz.agents.export_node")
from contentblitz.state import create_initial_state


def _base_state(**overrides):
    state = create_initial_state(
        final_response="Final compiled response.",
        export_metadata={
            "formats_requested": [],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        },
    )
    state.update(overrides)
    return state


def test_markdown_export_writes_path() -> None:
    state = _base_state(
        export_metadata={
            "formats_requested": ["markdown"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        }
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    assert metadata["export_paths"]["markdown"].endswith(".md")
    assert metadata["exported_at"]


def test_html_export_writes_path() -> None:
    state = _base_state(
        export_metadata={
            "formats_requested": ["html"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        }
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    assert metadata["export_paths"]["html"].endswith(".html")
    assert metadata["exported_at"]


def test_pdf_failure_marks_pdf_failed_safely(monkeypatch) -> None:
    def fake_export_content(content: str, format_name: str):
        if format_name == "pdf":
            raise RuntimeError("pdf export unavailable")
        return {"path": "exports/fallback.md"}

    monkeypatch.setattr(export_node_module, "export_content", fake_export_content)
    state = _base_state(
        export_metadata={
            "formats_requested": ["pdf"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        }
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]

    assert "pdf" not in metadata["export_paths"]
    assert metadata["export_status"]["pdf"] == "failed"
    assert "markdown" not in metadata["export_paths"]
    assert metadata["error_log"]
    assert metadata["error_log"][0]["format"] == "pdf"


def test_all_exports_fail_but_exported_at_is_still_set(monkeypatch) -> None:
    def always_fail(content: str, format_name: str):
        raise RuntimeError(f"{format_name} failed")

    monkeypatch.setattr(export_node_module, "export_content", always_fail)
    state = _base_state(
        export_metadata={
            "formats_requested": ["markdown", "html", "pdf"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        }
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]

    assert metadata["export_paths"] == {}
    assert metadata["exported_at"]
    assert len(metadata["error_log"]) >= 3


def test_final_response_is_unchanged() -> None:
    state = _base_state(
        final_response="Do not modify this response.",
        export_metadata={
            "formats_requested": ["markdown"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        },
    )
    updates = export_node_module.export_node(state)
    merged = {**state, **updates}

    assert merged["final_response"] == "Do not modify this response."


def test_empty_formats_requested_skips_export_but_sets_exported_at(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_export_content(content: str, format_name: str):
        calls["count"] += 1
        return {"path": f"exports/{format_name}"}

    monkeypatch.setattr(export_node_module, "export_content", fake_export_content)
    state = _base_state(
        export_metadata={
            "formats_requested": [],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        }
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]

    assert calls["count"] == 0
    assert metadata["export_paths"] == {}
    assert metadata["error_log"] == []
    assert metadata["exported_at"]


def test_validation_failure_marks_only_invalid_format_failed(monkeypatch) -> None:
    def fake_validate_markdown(*args, **kwargs):
        return {"valid": False, "warnings": [], "errors": ["bad markdown structure"]}

    def fake_validate_html(*args, **kwargs):
        return {"valid": True, "warnings": [], "errors": []}

    monkeypatch.setattr(
        export_node_module, "validate_markdown_export", fake_validate_markdown
    )
    monkeypatch.setattr(export_node_module, "validate_html_export", fake_validate_html)

    state = _base_state(
        export_requested=True,
        export_metadata={
            "formats_requested": ["markdown", "html"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]

    assert metadata["export_status"]["markdown"] == "failed"
    assert metadata["export_status"]["html"] == "completed"
    assert "markdown" not in metadata["export_paths"]
    assert metadata["export_paths"]["html"].endswith(".html")
    assert metadata["export_error_count"] == 1


def test_unsafe_export_path_is_rejected(monkeypatch) -> None:
    def fake_export_content(content: str, format_name: str):
        if format_name == "markdown":
            return {"path": "../outside.md"}
        return {"path": f"exports/{format_name}"}

    monkeypatch.setattr(export_node_module, "export_content", fake_export_content)
    state = _base_state(
        export_requested=True,
        export_metadata={
            "formats_requested": ["markdown"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]

    assert metadata["export_status"]["markdown"] == "failed"
    assert "markdown" not in metadata["export_paths"]
    assert metadata["export_error_count"] == 1


def test_validation_failure_adds_safe_status_message(monkeypatch) -> None:
    def fake_validate_markdown(*args, **kwargs):
        return {"valid": False, "warnings": [], "errors": ["invalid content"]}

    monkeypatch.setattr(
        export_node_module, "validate_markdown_export", fake_validate_markdown
    )
    state = _base_state(
        export_requested=True,
        export_metadata={
            "formats_requested": ["markdown"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
            "export_status": {},
        },
    )
    updates = export_node_module.export_node(state)
    metadata = updates["export_metadata"]
    messages = metadata.get("status_messages", [])

    assert metadata["export_status"]["markdown"] == "failed"
    assert isinstance(messages, list)
    assert any("failed validation" in str(message).lower() for message in messages)
