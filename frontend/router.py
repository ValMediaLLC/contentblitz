"""Simple page router for the Streamlit frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import streamlit as st

from frontend.pages import about, history, run_workflow


@dataclass(frozen=True)
class PageRoute:
    key: str
    title: str
    renderer: Callable[[], None]


ROUTES: List[PageRoute] = [
    PageRoute(key="run", title="Run Workflow", renderer=run_workflow.render),
    PageRoute(key="history", title="History", renderer=history.render),
    PageRoute(key="about", title="About", renderer=about.render),
]


def render_router() -> None:
    route_map: Dict[str, Callable[[], None]] = {route.title: route.renderer for route in ROUTES}
    selected_title = st.sidebar.radio("Navigation", list(route_map.keys()), index=0)
    route_map[selected_title]()

