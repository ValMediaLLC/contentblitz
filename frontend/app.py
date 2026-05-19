"""ContentBlitz Streamlit app entrypoint."""

from __future__ import annotations

import html
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

from contentblitz.ui.observability import build_observability_diagnostics
from frontend.config import FRONTEND_CONFIG
from frontend.router import render_router
from frontend.session import initialize_session_state
from frontend.theme import apply_frontend_theme

_OBSERVABILITY_STATUS_CLASS = {
    "enabled": "cbx-status-green",
    "disabled": "cbx-status-orange",
    "degraded": "cbx-status-red",
}


def _maybe_load_dotenv() -> None:
    if load_dotenv is None:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _status_pill_html(*, status_label: str, tone_class: str) -> str:
    safe_label = html.escape(str(status_label).strip() or "Disabled")
    allowed_tones = {"cbx-status-green", "cbx-status-orange", "cbx-status-red"}
    safe_tone = tone_class if tone_class in allowed_tones else "cbx-status-orange"
    return (
        f'<span class="cbx-status-pill {safe_tone}">'
        '<span class="cbx-status-pill-dot" aria-hidden="true"></span>'
        f'<span class="cbx-status-pill-text">{safe_label}</span>'
        "</span>"
    )


def _render_observability_status() -> None:
    """Render UI-safe observability diagnostics without blocking app execution."""
    try:
        diagnostics = build_observability_diagnostics()
    except Exception:  # pragma: no cover - defensive UI fallback
        diagnostics = {
            "status": "degraded",
            "status_label": "Degraded",
            "status_tone_class": "cbx-status-red",
            "tracing_enabled": False,
            "project_name": "ContentBlitz",
            "endpoint_host": "unknown",
            "last_trace_attempt_label": "Unavailable",
            "note": "Observability diagnostics are temporarily unavailable.",
            "dashboard_instruction": (
                "For trace details, review the LangSmith dashboard manually."
            ),
        }

    status = str(diagnostics.get("status", "")).strip().lower() or "disabled"
    tone_class = _OBSERVABILITY_STATUS_CLASS.get(
        status, str(diagnostics.get("status_tone_class", "cbx-status-orange"))
    )
    status_pill = _status_pill_html(
        status_label=str(diagnostics.get("status_label", "Disabled")),
        tone_class=tone_class,
    )
    tracing_enabled_text = (
        "true" if bool(diagnostics.get("tracing_enabled", False)) else "false"
    )
    project_name = html.escape(str(diagnostics.get("project_name", "ContentBlitz")))
    endpoint_host = html.escape(str(diagnostics.get("endpoint_host", "unknown")))
    last_trace_attempt_label = html.escape(
        str(diagnostics.get("last_trace_attempt_label", "Not requested"))
    )

    st.markdown(
        (
            "<div class='cbx-metric-grid'>"
            "<div class='cbx-metric-card'>"
            "<div class='cbx-metric-label'>Observability</div>"
            f"<div class='cbx-metric-value'>{status_pill}</div>"
            "</div>"
            "<div class='cbx-metric-card'>"
            "<div class='cbx-metric-label'>Tracing Enabled</div>"
            f"<div class='cbx-metric-value'>{tracing_enabled_text}</div>"
            "</div>"
            "<div class='cbx-metric-card'>"
            "<div class='cbx-metric-label'>Project</div>"
            f"<div class='cbx-metric-value'>{project_name}</div>"
            "</div>"
            "<div class='cbx-metric-card'>"
            "<div class='cbx-metric-label'>Endpoint Host</div>"
            f"<div class='cbx-metric-value'>{endpoint_host}</div>"
            "</div>"
            "<div class='cbx-metric-card'>"
            "<div class='cbx-metric-label'>Last Trace Attempt</div>"
            f"<div class='cbx-metric-value'>{last_trace_attempt_label}</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    note = str(diagnostics.get("note", "")).strip()
    if note:
        if status == "enabled":
            st.caption(note)
        elif status == "degraded":
            st.warning(note)
        else:
            st.info(note)

    dashboard_instruction = str(diagnostics.get("dashboard_instruction", "")).strip()
    if dashboard_instruction:
        st.caption(dashboard_instruction)


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
