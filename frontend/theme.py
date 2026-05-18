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
  --cbx-status-green-text: #166534;
  --cbx-status-green-bg: #dcfce7;
  --cbx-status-green-border: #86efac;
  --cbx-status-orange-text: #9a3412;
  --cbx-status-orange-bg: #ffedd5;
  --cbx-status-orange-border: #fdba74;
  --cbx-status-red-text: #991b1b;
  --cbx-status-red-bg: #fee2e2;
  --cbx-status-red-border: #fca5a5;
  --cbx-status-blue-text: #075985;
  --cbx-status-blue-bg: #e0f2fe;
  --cbx-status-blue-border: #7dd3fc;
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

.cbx-metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.55rem;
  margin: 0.35rem 0 0.75rem;
}

.cbx-metric-card {
  background: var(--cbx-surface);
  border: 1px solid var(--cbx-border);
  border-radius: 0.78rem;
  padding: 0.5rem 0.62rem;
  box-shadow: 0 2px 5px rgba(15, 23, 42, 0.04);
}

.cbx-metric-label {
  color: var(--cbx-subtext);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  line-height: 1.2;
}

.cbx-metric-value {
  color: var(--cbx-text);
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  line-height: 1.25;
  margin-top: 0.16rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cbx-summary-value {
  color: var(--cbx-text);
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  line-height: 1.25;
  margin-top: 0.16rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cbx-summary-dot {
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 999px;
  flex: 0 0 auto;
}

.cbx-summary-dot-idle {
  background: #94a3b8;
}

.cbx-summary-dot-running {
  background: #f59e0b;
  animation: cbxSummaryPulse 1s ease-in-out infinite;
}

.cbx-summary-dot-completed {
  background: #0d9488;
}

.cbx-summary-dot-failed {
  background: #dc2626;
}

.cbx-summary-dot-blue {
  background: #0284c7;
}

@keyframes cbxSummaryPulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.82); }
}

.cbx-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  width: fit-content;
  border: 1px solid currentColor;
  border-radius: 999px;
  padding: 0.08rem 0.45rem;
  font-family: "DM Sans", "Segoe UI", sans-serif;
  font-size: 0.76rem;
  font-weight: 700;
  line-height: 1.3;
  text-transform: none;
}

.cbx-status-pill-dot {
  width: 0.38rem;
  height: 0.38rem;
  border-radius: 999px;
  background: currentColor;
  opacity: 0.9;
}

.cbx-status-pill-text {
  line-height: 1.2;
}

.cbx-status-green {
  color: var(--cbx-status-green-text);
  background: var(--cbx-status-green-bg);
  border-color: var(--cbx-status-green-border);
}

.cbx-status-orange {
  color: var(--cbx-status-orange-text);
  background: var(--cbx-status-orange-bg);
  border-color: var(--cbx-status-orange-border);
}

.cbx-status-red {
  color: var(--cbx-status-red-text);
  background: var(--cbx-status-red-bg);
  border-color: var(--cbx-status-red-border);
}

.cbx-status-blue {
  color: var(--cbx-status-blue-text);
  background: var(--cbx-status-blue-bg);
  border-color: var(--cbx-status-blue-border);
}

.cbx-status-line {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin: 0.2rem 0 0.55rem;
}

.cbx-status-line-label {
  color: var(--cbx-text);
  font-size: 0.9rem;
  font-weight: 600;
}

.cbx-node-status-list {
  list-style: none;
  padding-left: 0;
  margin: 0.35rem 0 0.55rem;
}

.cbx-node-status-row {
  display: grid;
  grid-template-columns: 1.1rem minmax(0, 14rem) minmax(0, 1fr) auto 3.6rem;
  align-items: center;
  gap: 0.65rem;
  padding: 0.36rem 0.45rem;
  border-radius: 0.5rem;
  color: var(--cbx-text);
  font-size: 0.86rem;
  line-height: 1.45;
}

.cbx-node-row-running {
  background: rgba(255, 237, 213, 0.72);
}

.cbx-node-status-icon {
  color: var(--cbx-subtext);
  font-family: "DM Sans", "Segoe UI", sans-serif;
  font-size: 0.86rem;
  font-weight: 700;
  line-height: 1;
  text-align: center;
}

.cbx-node-row-running .cbx-node-status-icon {
  color: #b45309;
}

.cbx-node-icon-completed {
  color: #0d9488;
}

.cbx-node-icon-degraded {
  color: #f97316;
}

.cbx-node-icon-failed {
  color: #dc2626;
}

.cbx-node-name {
  color: var(--cbx-text);
  font-family: "DM Sans", "Segoe UI", sans-serif;
  font-weight: 500;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cbx-node-progress-track {
  width: 100%;
  height: 0.34rem;
  border-radius: 999px;
  background: var(--cbx-surface-soft);
  border: 1px solid var(--cbx-border);
  overflow: hidden;
  margin-top: 0;
}

.cbx-node-progress-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 220ms ease;
}

.cbx-node-status-badge {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  white-space: nowrap;
}

.cbx-node-elapsed {
  color: var(--cbx-subtext);
  font-family: "DM Sans", "Segoe UI", sans-serif;
  font-size: 0.74rem;
  font-weight: 600;
  text-align: right;
  white-space: nowrap;
}

.cbx-node-elapsed-running {
  color: #b45309;
}

.cbx-node-timer {
  color: var(--cbx-subtext);
  display: flex;
  justify-content: flex-end;
  font-family: "DM Sans", "Segoe UI", sans-serif;
  font-size: 0.78rem;
  font-weight: 700;
  margin-top: 0.35rem;
}

.cbx-node-status-green {
  background: linear-gradient(90deg, #22c55e, #16a34a);
}

.cbx-node-status-warning {
  background: linear-gradient(90deg, #fb923c, #f97316);
}

.cbx-node-status-red {
  background: linear-gradient(90deg, #ef4444, #dc2626);
}

.cbx-node-status-neutral {
  background: linear-gradient(90deg, #94a3b8, #64748b);
}

.cbx-node-status-running {
  background: linear-gradient(90deg, #f59e0b, #f97316, #f59e0b);
}

.cbx-node-progress-running {
  background-size: 240% 100%;
  animation:
    cbxNodeProgressGrow var(--cbx-node-duration, 1.4s) ease-in-out infinite,
    cbxNodeProgressShift 0.95s linear infinite;
}

@keyframes cbxNodeProgressShift {
  0% { background-position: 0% 0%; }
  100% { background-position: 100% 0%; }
}

@keyframes cbxNodeProgressGrow {
  0% { width: 18%; }
  80% { width: 92%; }
  100% { width: 92%; }
}
</style>
"""


def apply_frontend_theme() -> None:
    """Apply frontend visual styling and hide duplicate default navigation."""
    st.markdown(_FRONTEND_CSS, unsafe_allow_html=True)
