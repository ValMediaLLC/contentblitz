from __future__ import annotations

from pathlib import Path

from contentblitz.persistence.session_store import LocalSessionStore


def _record(run_id: str, updated_at: str) -> dict:
    return {
        "run_id": run_id,
        "session_id": "session-1",
        "created_at": updated_at,
        "updated_at": updated_at,
        "user_query": "q",
        "requested_outputs": ["blog"],
        "workflow_status": "success",
        "routing_decision": "content_strategist_node",
        "final_response": "ok",
    }


def test_creates_session_store_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "sessions"
    assert not base_dir.exists()
    _ = LocalSessionStore(base_dir=base_dir)
    assert base_dir.exists()
    assert base_dir.is_dir()


def test_save_and_load_workflow_run(tmp_path: Path) -> None:
    store = LocalSessionStore(base_dir=tmp_path / "sessions")
    run_id = store.save_run(_record("run-a", "2026-05-10T10:00:00+00:00"))
    loaded = store.load_run(run_id)
    assert loaded is not None
    assert loaded["run_id"] == "run-a"
    assert loaded["workflow_status"] == "success"


def test_lists_saved_runs_newest_first(tmp_path: Path) -> None:
    store = LocalSessionStore(base_dir=tmp_path / "sessions")
    store.save_run(_record("old", "2026-05-10T10:00:00+00:00"))
    store.save_run(_record("new", "2026-05-10T11:00:00+00:00"))
    listed = store.list_runs()
    assert [item["run_id"] for item in listed] == ["new", "old"]


def test_corrupt_json_file_is_skipped(tmp_path: Path) -> None:
    store = LocalSessionStore(base_dir=tmp_path / "sessions")
    store.save_run(_record("good", "2026-05-10T10:00:00+00:00"))
    (store.base_dir / "broken.json").write_text("{not-json", encoding="utf-8")
    listed = store.list_runs()
    assert [item["run_id"] for item in listed] == ["good"]


def test_missing_session_file_returns_none(tmp_path: Path) -> None:
    store = LocalSessionStore(base_dir=tmp_path / "sessions")
    assert store.load_run("missing-run") is None


def test_missing_run_id_is_rejected(tmp_path: Path) -> None:
    store = LocalSessionStore(base_dir=tmp_path / "sessions")
    try:
        store.save_run({"session_id": "x"})
    except ValueError:
        return
    raise AssertionError("Expected ValueError when run_id is missing.")
