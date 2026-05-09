"""ContentBlitz Streamlit app entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure `frontend.*` absolute imports work whether app is run from repo root
# (`streamlit run frontend/app.py`) or from inside `frontend/` (`streamlit run app.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.config import FRONTEND_CONFIG
from frontend.router import render_router
from frontend.session import initialize_session_state


def main() -> None:
    st.set_page_config(
        page_title=f"{FRONTEND_CONFIG.app_title} | {FRONTEND_CONFIG.page_title_suffix}",
        page_icon=FRONTEND_CONFIG.app_icon,
        layout="wide",
    )
    initialize_session_state()
    st.title(FRONTEND_CONFIG.app_title)
    st.caption(FRONTEND_CONFIG.page_title_suffix)
    render_router()


if __name__ == "__main__":
    main()
