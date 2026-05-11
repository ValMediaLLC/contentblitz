"""Frontend session-state helpers with isolated UI keys."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping

import streamlit as st

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
    st.session_state[KEY_INITIALIZED] = True


def set_last_result(result: Mapping[str, Any]) -> None:
    """Store a deep-copied workflow result in frontend session space."""
    st.session_state[KEY_LAST_RESULT] = deepcopy(dict(result))
    st.session_state[KEY_LAST_ERROR] = None


def set_last_error(message: str) -> None:
    st.session_state[KEY_LAST_ERROR] = str(message).strip()


def set_execution_status(status: str) -> None:
    normalized = str(status).strip().lower() or "idle"
    st.session_state[KEY_EXECUTION_STATUS] = normalized


def get_execution_status() -> str:
    return str(st.session_state.get(KEY_EXECUTION_STATUS, "idle")).strip().lower() or "idle"


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
    return str(st.session_state.get(KEY_LAST_ERROR, "")).strip()


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
    st.session_state[KEY_STATUS_MESSAGES] = [str(item).strip() for item in messages if str(item).strip()]


def get_status_messages() -> List[str]:
    value = st.session_state.get(KEY_STATUS_MESSAGES, [])
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
