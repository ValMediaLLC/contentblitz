"""Local JSON-backed persistence store for workflow session runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_SESSION_DIR = ".contentblitz_sessions"
SESSION_DIR_ENV_VAR = "CONTENTBLITZ_SESSION_DIR"


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def resolve_session_store_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser()
    env_value = _safe_text(os.getenv(SESSION_DIR_ENV_VAR, ""))
    if env_value:
        return Path(env_value).expanduser()
    return Path(DEFAULT_SESSION_DIR)


class LocalSessionStore:
    """Simple local file-backed store with one JSON file per run."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = resolve_session_store_dir(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        safe_run_id = _safe_text(run_id)
        return self.base_dir / f"{safe_run_id}.json"

    def save_run(self, run_payload: Mapping[str, Any]) -> str:
        run_id = _safe_text(run_payload.get("run_id"))
        if not run_id:
            raise ValueError("run_id is required for persistence.")

        path = self._run_path(run_id)
        temp_path = path.with_suffix(".tmp")
        serialized = json.dumps(
            dict(run_payload), indent=2, ensure_ascii=True, sort_keys=True
        )
        temp_path.write_text(serialized, encoding="utf-8")
        temp_path.replace(path)
        return run_id

    def load_run(self, run_id: str) -> Dict[str, Any] | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        if _safe_text(payload.get("run_id")) != _safe_text(run_id):
            return None
        return payload

    def list_runs(self, *, limit: int | None = None) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in self.base_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            run_id = _safe_text(payload.get("run_id"))
            if not run_id:
                continue
            records.append(payload)

        def _sort_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
            updated = _safe_text(item.get("updated_at"))
            created = _safe_text(item.get("created_at"))
            run_id = _safe_text(item.get("run_id"))
            return (updated, created, run_id)

        records.sort(key=_sort_key, reverse=True)
        if isinstance(limit, int) and limit > 0:
            return records[:limit]
        return records
