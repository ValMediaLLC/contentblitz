"""Page describing frontend architecture boundaries."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("About ContentBlitz")
    st.markdown(
        (
            "<p class='cbx-page-intro'>"
            "ContentBlitz is a workflow-oriented content studio that combines "
            "research, multi-format drafting, and export-ready outputs in one "
            "orchestration-first UI."
            "</p>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<section class='cbx-about-grid'>"
            "<article class='cbx-about-card'>"
            "<p class='cbx-about-card-title'>Research-backed content</p>"
            "<p class='cbx-about-card-copy'>"
            "Generate drafts grounded in structured research and cited sources."
            "</p>"
            "</article>"
            "<article class='cbx-about-card'>"
            "<p class='cbx-about-card-title'>LinkedIn and blog formats</p>"
            "<p class='cbx-about-card-copy'>"
            "Create channel-specific outputs for social distribution and "
            "long-form publishing."
            "</p>"
            "</article>"
            "<article class='cbx-about-card'>"
            "<p class='cbx-about-card-title'>Image concepts</p>"
            "<p class='cbx-about-card-copy'>"
            "Explore visual concepts that align with campaign positioning and "
            "narrative."
            "</p>"
            "</article>"
            "<article class='cbx-about-card'>"
            "<p class='cbx-about-card-title'>Export-ready deliverables</p>"
            "<p class='cbx-about-card-copy'>"
            "Package outputs into PDF, DOCX, Markdown, and HTML when exports "
            "are requested."
            "</p>"
            "</article>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        "\n".join(
            [
                "- UI consumes orchestration outputs from the existing "
                "LangGraph pipeline.",
                "- No provider calls are made directly from Streamlit components.",
                "- No orchestration business logic is duplicated in the frontend.",
                "- Session keys are isolated under a frontend-specific prefix.",
            ]
        )
    )
