"""Page for viewing frontend run history."""

from __future__ import annotations

import streamlit as st

from frontend.session import get_run_history


def render() -> None:
    st.header("Run History")
    history = get_run_history()
    if not history:
        st.info("No runs yet in this session.")
        return

    for item in history:
        timestamp = str(item.get("timestamp_utc", "")).strip()
        query = str(item.get("user_query", "")).strip()
        outputs = item.get("requested_outputs", [])
        status = str(item.get("workflow_status", "")).strip()
        st.markdown(
            f"- `{timestamp}` | status: `{status or 'unknown'}` | outputs: `{outputs}` | query: {query}"
        )

