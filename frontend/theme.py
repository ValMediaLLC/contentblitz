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

header[data-testid="stHeader"],
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] {
  display: none !important;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f9fcff 0%, #f2f7fc 100%);
}

section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
  display: none;
}

section[data-testid="stMain"] > div[data-testid="stMainBlockContainer"] {
  padding-top: 1.1rem;
  padding-bottom: 3.4rem;
  max-width: 1140px;
}

.cbx-page-intro {
  color: #475569;
  font-size: 0.95rem;
  line-height: 1.5;
  margin: 0.15rem 0 0.52rem;
}

div.st-key-cbx_run_form_card {
  margin-top: 0.28rem;
  margin-bottom: 0.96rem;
}

.cbx-form-header {
  color: #1e293b;
  font-family: "DM Sans", "Segoe UI", sans-serif;
  font-size: 0.94rem;
  font-weight: 600;
  margin: 0.02rem 0 0.16rem;
}

.cbx-form-helper {
  color: #64748b;
  font-size: 0.77rem !important;
  font-style: italic;
  line-height: 1.4;
  margin: 0 0 0.46rem;
}

div.st-key-cbx_run_form_card > div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 1.05rem;
  border: 1px solid #c6d7e8;
  background:
    linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.93) 0%,
      rgba(247, 251, 255, 0.95) 100%
    );
  box-shadow:
    0 14px 32px rgba(15, 23, 42, 0.085),
    inset 0 1px 0 rgba(255, 255, 255, 0.8);
}

section[data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"] {
  box-shadow: 0 9px 22px rgba(15, 23, 42, 0.06);
}

div.st-key-cbx_run_form_card label[data-testid="stWidgetLabel"] p {
  color: #1e293b;
  font-weight: 600;
}

div.st-key-cbx_query_input textarea {
  border-radius: 0.8rem;
  line-height: 1.5;
  background: rgba(248, 252, 255, 0.92);
  border-color: #cfdceb !important;
}

div.st-key-cbx_query_input {
  margin-bottom: 0.18rem;
}

div.st-key-cbx_quick_start {
  margin-top: 0.16rem;
  margin-bottom: 0.46rem;
}

div.st-key-cbx_quick_start div[role="radiogroup"] {
  gap: 0.28rem;
}

div.st-key-cbx_quick_start label,
div.st-key-cbx_quick_start button {
  border-radius: 999px !important;
  border: 1px solid #c6d7e8 !important;
  background: linear-gradient(180deg, #ffffff 0%, #f2f8ff 100%) !important;
  color: #1e293b !important;
  font-family: "DM Sans", "Segoe UI", sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  cursor: pointer !important;
  transition:
    transform 0.14s ease,
    box-shadow 0.18s ease,
    border-color 0.18s ease !important;
}

div.st-key-cbx_quick_start label:hover,
div.st-key-cbx_quick_start button:hover {
  border-color: #7fb6de !important;
  box-shadow: 0 8px 15px rgba(15, 23, 42, 0.1);
  transform: translateY(-1px);
}

div.st-key-cbx_quick_start label:focus-within,
div.st-key-cbx_quick_start button:focus-visible {
  outline: none !important;
  border-color: #0d9488 !important;
  box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.18) !important;
}

div.st-key-cbx_quick_start label:has(input:checked),
div.st-key-cbx_quick_start button[aria-pressed="true"] {
  background:
    linear-gradient(120deg, var(--cbx-accent), var(--cbx-accent-deep)) !important;
  color: #f8fafc !important;
  border-color: transparent !important;
  box-shadow: 0 9px 17px rgba(3, 105, 161, 0.26);
  transform: translateY(-1px);
}

div.st-key-cbx_run_submit {
  margin-top: 0.22rem;
}

div.st-key-cbx_run_submit button[kind="primary"] {
  min-height: 2.5rem !important;
  padding-inline: 1rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
  box-shadow: 0 10px 20px rgba(3, 105, 161, 0.24);
}

div.st-key-cbx_run_submit button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 24px rgba(3, 105, 161, 0.3);
}

div.st-key-cbx_run_submit button[kind="primary"]:focus-visible {
  outline: none !important;
  box-shadow:
    0 0 0 2px rgba(255, 255, 255, 0.78),
    0 0 0 4px rgba(13, 148, 136, 0.35),
    0 12px 22px rgba(3, 105, 161, 0.25) !important;
}

div[data-testid="stForm"] {
  border-radius: 1rem;
  border: 1px solid var(--cbx-border);
  background: var(--cbx-surface);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
}

div.st-key-cbx_route_nav {
  margin: 0.38rem 0 1.28rem;
}

.cbx-idle-examples {
  margin-top: 0.15rem;
}

.cbx-idle-examples-title {
  color: #64748b;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  margin: 0 0 0.48rem;
}

.cbx-idle-examples-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.56rem;
}

.cbx-idle-example-card {
  border: 1px solid #d7e3ef;
  border-radius: 0.84rem;
  padding: 0.64rem 0.74rem;
  background: rgba(255, 255, 255, 0.5);
}

.cbx-idle-example-title {
  color: #1e293b;
  font-size: 0.83rem;
  font-weight: 600;
  margin: 0 0 0.2rem;
}

.cbx-idle-example-copy {
  color: #64748b;
  font-size: 0.78rem;
  line-height: 1.35;
  margin: 0;
}

@media (max-width: 900px) {
  .cbx-idle-examples-grid {
    grid-template-columns: 1fr;
  }
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

.cbx-perf-table-wrap {
  margin: 0.2rem 0 0.66rem;
  border: 1px solid #d5e2ef;
  border-radius: 0.84rem;
  background: rgba(255, 255, 255, 0.52);
  overflow-x: auto;
}

.cbx-perf-table {
  width: 100%;
  border-collapse: collapse;
  min-width: 760px;
}

.cbx-perf-table th,
.cbx-perf-table td {
  padding: 0.42rem 0.56rem;
  border-bottom: 1px solid #e3edf6;
  text-align: left;
  vertical-align: middle;
  font-size: 0.79rem;
  line-height: 1.35;
  white-space: nowrap;
}

.cbx-perf-table th {
  color: #475569;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  text-transform: uppercase;
  background: rgba(241, 247, 253, 0.9);
}

.cbx-perf-table tbody tr:last-child td {
  border-bottom: none;
}

.cbx-perf-table .cbx-perf-cell-mono {
  font-family: "JetBrains Mono", "Consolas", "Segoe UI Mono", monospace;
  font-variant-numeric: tabular-nums;
}

.cbx-perf-table .cbx-perf-cell-status .cbx-status-pill {
  font-size: 0.72rem;
}

.cbx-perf-table .cbx-perf-cell-provider,
.cbx-perf-table .cbx-perf-cell-model {
  max-width: 15ch;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cbx-perf-table .cbx-perf-cell-muted {
  color: #64748b;
}

.cbx-cache-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 2.75rem;
  border-radius: 999px;
  border: 1px solid transparent;
  padding: 0.06rem 0.42rem;
  font-size: 0.69rem;
  font-weight: 700;
  line-height: 1.2;
  text-transform: lowercase;
}

.cbx-cache-badge-hit {
  color: #166534;
  background: rgba(220, 252, 231, 0.8);
  border-color: rgba(34, 197, 94, 0.4);
}

.cbx-cache-badge-miss {
  color: #9a3412;
  background: rgba(255, 237, 213, 0.78);
  border-color: rgba(251, 146, 60, 0.45);
}

.cbx-cache-badge-na {
  color: #64748b;
  background: rgba(226, 232, 240, 0.72);
  border-color: rgba(148, 163, 184, 0.42);
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

.cbx-status-neutral {
  color: #475569;
  background: rgba(148, 163, 184, 0.2);
  border-color: rgba(100, 116, 139, 0.35);
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
  padding-inline-start: 0;
  margin: 0.3rem 0 0.3rem;
}

.cbx-node-status-row {
  display: grid;
  grid-template-columns: 0.9rem minmax(0, 14rem) minmax(0, 1fr) auto minmax(5ch, auto);
  align-items: center;
  gap: 0.56rem;
  border-radius: 0.58rem;
  border: 1px solid transparent;
  color: var(--cbx-text);
  font-size: 0.86rem;
  line-height: 1.45;
  transition: border-color 0.16s ease, box-shadow 0.18s ease;
}

/* Higher specificity keeps row spacing from Streamlit's global li rule. */
.cbx-node-status-list > li.cbx-node-status-row {
  margin: 0.3em 0.7em 0.3em 0.7em;
  padding: 0.15rem 0.38rem 0.15rem 0.38rem;
}

.cbx-node-status-row + .cbx-node-status-row {
  margin-top: 0;
}

.cbx-node-status-panel {
  border: 1px solid #d5e2ef;
  border-radius: 0.86rem;
  background: linear-gradient(180deg, #f8fcff 0%, #f3f8fd 100%);
  padding: 1.42rem 0.54rem 1.22rem 0.42rem;
  margin-bottom: 1.66rem;
}

.cbx-node-status-panel-secondary {
  margin-top: 0.18rem;
  border-style: dashed;
  background: linear-gradient(180deg, #f7fafd 0%, #f1f5f9 100%);
}

.cbx-node-panel-running {
  animation: cbxNodePanelPulse 1.3s ease-in-out infinite;
}

@keyframes cbxNodePanelPulse {
  0%, 100% { box-shadow: 0 0 0 rgba(14, 165, 163, 0); }
  50% { box-shadow: 0 0 0 2px rgba(14, 165, 163, 0.08); }
}

.cbx-node-row-active {
  border-color: rgba(14, 165, 163, 0.38);
  box-shadow: 0 5px 14px rgba(14, 165, 163, 0.12);
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

.cbx-node-status-dot {
  display: inline-block;
  width: 0.56rem;
  height: 0.56rem;
  border-radius: 999px;
  background: currentColor;
  opacity: 0.9;
}

.cbx-node-row-running .cbx-node-status-icon {
  color: #b45309;
}

.cbx-node-row-running .cbx-node-status-dot {
  animation: cbxSummaryPulse 1s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .cbx-node-row-running .cbx-node-status-dot {
    animation: none;
  }
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
  display: flex;
  flex-direction: column;
  gap: 0.05rem;
  min-width: 0;
}

.cbx-node-title {
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cbx-node-title-friendly {
  font-size: 0.90rem;
  line-height: 1.3;
}

.cbx-node-meta {
  color: #64748b;
  font-family: "JetBrains Mono", "Consolas", monospace;
  font-size: 0.69rem;
  line-height: 1.15;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cbx-node-progress-track {
  --cbx-node-bar-height: 0.38rem;
  width: 100%;
  height: var(--cbx-node-bar-height);
  min-height: var(--cbx-node-bar-height);
  max-height: var(--cbx-node-bar-height);
  border-radius: 999px;
  background: var(--cbx-surface-soft);
  border: 1px solid var(--cbx-border);
  overflow: hidden;
  margin-top: 0;
  align-self: center;
  box-sizing: border-box;
}

.cbx-node-progress-fill {
  display: block;
  height: 100%;
  min-height: 100%;
  max-height: 100%;
  border-radius: 999px;
  transition: width 220ms ease;
  box-sizing: border-box;
}

.cbx-node-status-badge {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  white-space: nowrap;
  min-height: 1.62rem;
}

.cbx-node-status-badge .cbx-status-orange {
  padding: 0.2rem 0.66rem;
}

.cbx-node-status-badge-hidden {
  min-width: 0;
  width: 0;
  padding: 0;
  margin: 0;
  overflow: hidden;
}

.cbx-node-elapsed {
  color: var(--cbx-subtext);
  font-family: "JetBrains Mono", "Consolas", "Segoe UI Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-size: 0.74rem;
  font-weight: 600;
  text-align: right;
  white-space: nowrap;
  padding-right: 0.14rem;
}

.cbx-node-elapsed-running {
  color: #b45309;
}

.cbx-node-timer {
  color: var(--cbx-subtext);
  display: flex;
  justify-content: flex-end;
  font-family: "JetBrains Mono", "Consolas", "Segoe UI Mono", monospace;
  font-variant-numeric: tabular-nums;
  font-size: 0.78rem;
  font-weight: 700;
  margin-top: 0.2rem;
  margin-right: 1.2rem;
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

.cbx-history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.56rem;
  margin: 0.3rem 0 0.8rem;
}

.cbx-history-card,
.cbx-history-timeline-card {
  border: 1px solid #d8e3ef;
  border-radius: 0.84rem;
  padding: 0.58rem 0.68rem;
  background: rgba(255, 255, 255, 0.52);
}

.cbx-history-timeline-card {
  margin: 0.24rem 0 0.12rem;
}

.cbx-history-card-meta {
  color: #64748b;
  font-size: 0.75rem;
  margin: 0.32rem 0 0.18rem;
}

.cbx-history-card-query {
  color: #1e293b;
  font-size: 0.83rem;
  line-height: 1.38;
  margin: 0;
}

.cbx-about-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 0.58rem;
  margin: 0.26rem 0 0.82rem;
}

.cbx-about-card {
  border: 1px solid #d8e3ef;
  border-radius: 0.9rem;
  padding: 0.7rem 0.78rem;
  background: rgba(255, 255, 255, 0.56);
}

.cbx-about-card-title {
  color: #0f172a;
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
  font-size: 0.92rem;
  font-weight: 600;
  margin: 0 0 0.24rem;
}

.cbx-about-card-copy {
  color: #475569;
  font-size: 0.8rem;
  line-height: 1.4;
  margin: 0;
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
