"""ContentBlitz Streamlit app entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - defensive import guard
    load_dotenv = None

# Ensure `frontend.*` absolute imports work whether app is run from repo root
# (`streamlit run frontend/app.py`) or from inside `frontend/` (`streamlit run app.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.config import FRONTEND_CONFIG
from frontend.router import render_router
from frontend.session import initialize_session_state
from frontend.theme import apply_frontend_theme


def _maybe_load_dotenv() -> None:
    if load_dotenv is None:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def main() -> None:
    _maybe_load_dotenv()
    st.set_page_config(
        page_title=f"{FRONTEND_CONFIG.app_title} | {FRONTEND_CONFIG.page_title_suffix}",
        page_icon=FRONTEND_CONFIG.app_icon,
        layout="wide",
    )
    st.set_option("client.showSidebarNavigation", False)
    apply_frontend_theme()
    initialize_session_state()

    logo_col, heading_col = st.columns([1.0, 2.6], vertical_alignment="center")
    with logo_col:
        st.image(
            str(PROJECT_ROOT / FRONTEND_CONFIG.logo_path),
            use_container_width=True,
        )
    with heading_col:
        st.markdown(
            (
                f"### {FRONTEND_CONFIG.app_title}\n"
                f"<p class='cbx-hero-subtitle'>{FRONTEND_CONFIG.page_title_suffix}</p>"
            ),
            unsafe_allow_html=True,
        )

    st.logo(
        str(PROJECT_ROOT / FRONTEND_CONFIG.logo_path),
        icon_image=str(PROJECT_ROOT / FRONTEND_CONFIG.logo_icon_path),
        size="large",
    )
    render_router()


if __name__ == "__main__":
    main()
