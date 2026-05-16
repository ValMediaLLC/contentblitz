"""Page describing frontend architecture boundaries."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("About This UI Shell")
    st.markdown(
        "\n".join(
            [
                "- UI consumes orchestration outputs from the existing LangGraph pipeline.",
                "- No provider calls are made directly from Streamlit components.",
                "- No orchestration business logic is duplicated in the frontend.",
                "- Session keys are isolated under a frontend-specific prefix.",
            ]
        )
    )
