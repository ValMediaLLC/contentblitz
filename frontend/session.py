"""Frontend session-state helpers with isolated UI keys."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping
from uuid import uuid4

import streamlit as st
from contentblitz.persistence.serialization import (
    deserialize_workflow_run,
    serialize_workflow_run,
    to_run_summary,
)
from contentblitz.persistence.session_store import LocalSessionStore

SESSION_PREFIX = "cbx_ui_"
KEY_INITIALIZED = f"{SESSION_PREFIX}initialized"
KEY_LAST_RESULT = f"{SESSION_PREFIX}last_result"
KEY_LAST_ERROR = f"{SESSION_PREFIX}last_error"
KEY_RUN_HISTORY = f"{SESSION_PREFIX}run_history"
KEY_EXECUTION_STATUS = f"{SESSION_PREFIX}execution_status"
KEY_LAST_SUBMISSION = f"{SESSION_PREFIX}last_submission"
KEY_PROGRESS_EVENTS = f"{SESSION_PREFIX}progress_events"
KEY_NODE_STATUSES = f"{SESSION_PREFIX}node_statuses"
KEY_STATUS_MESSAGES = f"{SESSION_PREFIX}status_messages"
KEY_UI_SESSION_ID = f"{SESSION_PREFIX}ui_session_id"
KEY_PERSISTENCE_MESSAGES = f"{SESSION_PREFIX}persistence_messages"


@dataclass(frozen=True)
class RunHistoryItem:
    timestamp_utc: str
    user_query: str
    requested_outputs: List[str]
    workflow_status: str


def initialize_session_state() -> None:
    """Ensure frontend-only session keys exist for current browser session."""
    if st.session_state.get(KEY_INITIALIZED, False):
        return

    st.session_state[KEY_LAST_RESULT] = None
    st.session_state[KEY_LAST_ERROR] = None
    st.session_state[KEY_RUN_HISTORY] = []
    st.session_state[KEY_EXECUTION_STATUS] = "idle"
    st.session_state[KEY_LAST_SUBMISSION] = {}
    st.session_state[KEY_PROGRESS_EVENTS] = []
    st.session_state[KEY_NODE_STATUSES] = {}
    st.session_state[KEY_STATUS_MESSAGES] = []
    st.session_state[KEY_UI_SESSION_ID] = uuid4().hex
    st.session_state[KEY_PERSISTENCE_MESSAGES] = []
    st.session_state[KEY_INITIALIZED] = True


def get_ui_session_id() -> str:
    value = st.session_state.get(KEY_UI_SESSION_ID, "")
    cleaned = str(value).strip()
    if cleaned:
        return cleaned
    generated = uuid4().hex
    st.session_state[KEY_UI_SESSION_ID] = generated
    return generated


def set_last_result(result: Mapping[str, Any]) -> None:
    """Store a deep-copied workflow result in frontend session space."""
    st.session_state[KEY_LAST_RESULT] = deepcopy(dict(result))
    st.session_state[KEY_LAST_ERROR] = None


def set_last_error(message: str) -> None:
    raw = message
    if raw is None:
        st.session_state[KEY_LAST_ERROR] = ""
        return
    cleaned = str(raw).strip()
    if cleaned.lower() in {"none", "null"}:
        cleaned = ""
    st.session_state[KEY_LAST_ERROR] = cleaned


def set_execution_status(status: str) -> None:
    normalized = str(status).strip().lower() or "idle"
    st.session_state[KEY_EXECUTION_STATUS] = normalized


def get_execution_status() -> str:
    return (
        str(st.session_state.get(KEY_EXECUTION_STATUS, "idle")).strip().lower()
        or "idle"
    )


def set_last_submission(submission: Mapping[str, Any]) -> None:
    st.session_state[KEY_LAST_SUBMISSION] = deepcopy(dict(submission))


def get_last_submission() -> Dict[str, Any]:
    value = st.session_state.get(KEY_LAST_SUBMISSION, {})
    if isinstance(value, dict):
        return deepcopy(value)
    return {}


def add_history_entry(
    *,
    user_query: str,
    requested_outputs: List[str],
    workflow_status: str,
) -> None:
    history = list(st.session_state.get(KEY_RUN_HISTORY, []))
    item = RunHistoryItem(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        user_query=str(user_query).strip(),
        requested_outputs=[str(value).strip() for value in requested_outputs],
        workflow_status=str(workflow_status).strip(),
    )
    history.insert(0, asdict(item))
    st.session_state[KEY_RUN_HISTORY] = history[:20]


def get_last_result() -> Dict[str, Any] | None:
    value = st.session_state.get(KEY_LAST_RESULT)
    if isinstance(value, dict):
        return deepcopy(value)
    return None


def get_last_error() -> str:
    value = st.session_state.get(KEY_LAST_ERROR, "")
    if value is None:
        return ""
    cleaned = str(value).strip()
    if cleaned.lower() in {"none", "null"}:
        return ""
    return cleaned


def get_run_history() -> List[Dict[str, Any]]:
    history = st.session_state.get(KEY_RUN_HISTORY, [])
    if not isinstance(history, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in history:
        if isinstance(item, dict):
            cleaned.append(deepcopy(item))
    return cleaned


def set_progress_events(events: List[Mapping[str, Any]]) -> None:
    cleaned: list[dict[str, Any]] = []
    for item in events:
        if isinstance(item, Mapping):
            cleaned.append(deepcopy(dict(item)))
    st.session_state[KEY_PROGRESS_EVENTS] = cleaned


def get_progress_events() -> List[Dict[str, Any]]:
    value = st.session_state.get(KEY_PROGRESS_EVENTS, [])
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            cleaned.append(deepcopy(item))
    return cleaned


def set_node_statuses(statuses: Mapping[str, str]) -> None:
    st.session_state[KEY_NODE_STATUSES] = deepcopy(dict(statuses))


def get_node_statuses() -> Dict[str, str]:
    value = st.session_state.get(KEY_NODE_STATUSES, {})
    if isinstance(value, dict):
        return deepcopy({str(key): str(val) for key, val in value.items()})
    return {}


def set_status_messages(messages: List[str]) -> None:
    cleaned_messages: list[str] = []
    for item in messages:
        if item is None:
            continue
        cleaned = str(item).strip()
        if not cleaned or cleaned.lower() in {"none", "null"}:
            continue
        cleaned_messages.append(cleaned)
    st.session_state[KEY_STATUS_MESSAGES] = cleaned_messages


def get_status_messages() -> List[str]:
    value = st.session_state.get(KEY_STATUS_MESSAGES, [])
    if not isinstance(value, list):
        return []
    cleaned_messages: list[str] = []
    for item in value:
        if item is None:
            continue
        cleaned = str(item).strip()
        if not cleaned or cleaned.lower() in {"none", "null"}:
            continue
        cleaned_messages.append(cleaned)
    return cleaned_messages


def add_persistence_message(message: str) -> None:
    cleaned = str(message or "").strip()
    if not cleaned or cleaned.lower() in {"none", "null"}:
        return
    existing = st.session_state.get(KEY_PERSISTENCE_MESSAGES, [])
    if not isinstance(existing, list):
        existing = []
    existing.append(cleaned)
    st.session_state[KEY_PERSISTENCE_MESSAGES] = existing[-20:]


def get_persistence_messages() -> List[str]:
    raw = st.session_state.get(KEY_PERSISTENCE_MESSAGES, [])
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        message = str(item or "").strip()
        if not message or message.lower() in {"none", "null"}:
            continue
        cleaned.append(message)
    return cleaned


def clear_persistence_messages() -> None:
    st.session_state[KEY_PERSISTENCE_MESSAGES] = []


def _get_store() -> LocalSessionStore:
    return LocalSessionStore()


def save_persisted_run(
    *,
    result: Mapping[str, Any],
    last_submission: Mapping[str, Any],
    progress_events: List[Mapping[str, Any]],
    node_statuses: Mapping[str, str],
    status_messages: List[str],
) -> str | None:
    """
    Persist a workflow run safely to local storage.

    Returns run_id when persisted successfully, else None.
    """
    try:
        serializable_result = dict(result)
        serializable_result["ui_node_statuses"] = deepcopy(dict(node_statuses))
        serializable_result["status_messages"] = deepcopy(list(status_messages))
        serializable_result["ui_selected_options"] = deepcopy(dict(last_submission))
        record = serialize_workflow_run(
            result_state=serializable_result,
            ui_selected_options=last_submission,
            progress_events=progress_events,
            status_messages=status_messages,
            session_id=get_ui_session_id(),
        )
        run_id = _get_store().save_run(record)
        return run_id
    except Exception:
        add_persistence_message(
            "Workflow run completed, but local persistence save failed."
        )
        return None


def list_persisted_run_summaries(*, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        records = _get_store().list_runs(limit=limit)
    except Exception:
        add_persistence_message("Unable to list saved workflow runs.")
        return []
    summaries: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        summaries.append(to_run_summary(item))
    return summaries


def load_persisted_run(run_id: str) -> Dict[str, Any] | None:
    safe_run_id = str(run_id or "").strip()
    if not safe_run_id:
        return None
    try:
        record = _get_store().load_run(safe_run_id)
    except Exception:
        add_persistence_message("Unable to load the selected workflow run.")
        return None
    if not isinstance(record, dict):
        return None
    try:
        return deserialize_workflow_run(record)
    except Exception:
        add_persistence_message("Saved workflow run is unavailable or corrupted.")
        return None


def restore_persisted_run(run_id: str) -> tuple[bool, str]:
    restored = load_persisted_run(run_id)
    if not isinstance(restored, dict):
        return False, "Saved workflow run could not be restored."

    set_last_result(restored)
    workflow_status = (
        str(
            restored.get("ui_workflow_status", "")
            or restored.get("workflow_status", "")
        )
        .strip()
        .lower()
    )
    set_execution_status(workflow_status or "completed")
    set_progress_events(
        [
            item
            for item in restored.get("ui_progress_events", [])
            if isinstance(item, Mapping)
        ]
    )
    set_node_statuses(
        {
            str(key): str(value)
            for key, value in dict(restored.get("ui_node_statuses", {})).items()
        }
    )
    set_status_messages(
        [
            *[
                item
                for item in restored.get("status_messages", [])
                if isinstance(item, str)
            ],
            "Restored saved workflow run.",
        ]
    )
    set_last_submission(
        dict(restored.get("ui_selected_options", {}))
        if isinstance(restored.get("ui_selected_options", {}), Mapping)
        else {}
    )
    set_last_error("")
    return True, "Saved workflow run restored."
