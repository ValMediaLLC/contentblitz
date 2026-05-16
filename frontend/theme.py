"""Theme and style helpers for the Streamlit frontend."""

from __future__ import annotations

import streamlit as st

_FRONTEND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
  --cbx-bg: #f4f8fc;
  --cbx-surface: #ffffff;
  --cbx-surface-soft: #eaf1f9;
  --cbx-border: #d4deea;
  --cbx-text: #0f172a;
  --cbx-subtext: #475569;
  --cbx-accent: #0d9488;
  --cbx-accent-deep: #0369a1;
}

html, body, [class*="css"] {
  font-family: "DM Sans", "Segoe UI", sans-serif !important;
  color: var(--cbx-text);
}

h1, h2, h3, h4, h5, h6 {
  font-family: "Space Grotesk", "Segoe UI", sans-serif !important;
  letter-spacing: -0.01em;
  color: var(--cbx-text);
}

section[data-testid="stMain"] {
  background:
    radial-gradient(circle at 8% 4%, rgba(14, 165, 163, 0.14), transparent 26%),
    radial-gradient(circle at 92% 2%, rgba(3, 105, 161, 0.12), transparent 24%),
    var(--cbx-bg);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f9fcff 0%, #f2f7fc 100%);
}

section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
  display: none;
}

section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 1180px;
}

.cbx-hero-subtitle {
  margin: 0.2rem 0 0;
  color: var(--cbx-subtext);
  font-size: 0.98rem;
}

div[data-testid="stForm"] {
  border-radius: 1rem;
  border: 1px solid var(--cbx-border);
  background: var(--cbx-surface);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
}

div.st-key-cbx_route_nav {
  margin: 0.2rem 0 1.15rem;
}

div.st-key-cbx_route_nav div[role="radiogroup"] {
  background: var(--cbx-surface-soft);
  border: 1px solid var(--cbx-border);
  border-radius: 999px;
  padding: 0.28rem;
  gap: 0.22rem;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

div.st-key-cbx_route_nav label {
  border-radius: 999px !important;
  transition: transform 0.18s ease, background-color 0.22s ease,
    color 0.22s ease, box-shadow 0.22s ease;
  color: #334155 !important;
  font-family: "Space Grotesk", "Segoe UI", sans-serif !important;
  font-weight: 600 !important;
}

div.st-key-cbx_route_nav label:hover {
  background-color: #dce7f3 !important;
  transform: translateY(-1px);
}

div.st-key-cbx_route_nav label:has(input:checked) {
  background: linear-gradient(120deg, var(--cbx-accent), var(--cbx-accent-deep));
  color: #f8fafc !important;
  box-shadow: 0 8px 18px rgba(3, 105, 161, 0.26);
}

button[kind="primary"] {
  border-radius: 0.85rem !important;
  background: linear-gradient(120deg, var(--cbx-accent), var(--cbx-accent-deep));
  border: none !important;
  transition: transform 0.16s ease, box-shadow 0.2s ease;
}

button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 20px rgba(3, 105, 161, 0.28);
}
</style>
"""


def apply_frontend_theme() -> None:
    """Apply frontend visual styling and hide duplicate default navigation."""
    st.markdown(_FRONTEND_CSS, unsafe_allow_html=True)
