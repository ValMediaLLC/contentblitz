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

ROUTE_LABELS: Dict[str, str] = {
    "run": ":material/rocket_launch: Run Workflow",
    "history": ":material/history: History",
    "about": ":material/info: About",
}


def render_router() -> None:
    route_by_key: Dict[str, PageRoute] = {route.key: route for route in ROUTES}
    route_keys = [route.key for route in ROUTES]

    selected_key = st.segmented_control(
        label="Navigation",
        options=route_keys,
        default=route_keys[0],
        format_func=lambda route_key: ROUTE_LABELS.get(route_key, route_key.title()),
        key="cbx_route_nav",
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
    )
    active_key = selected_key if selected_key in route_by_key else route_keys[0]
    route_by_key[active_key].renderer()
