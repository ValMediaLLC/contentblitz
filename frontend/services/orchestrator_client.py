"""Thin frontend adapter over ContentBlitz orchestration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph

_GRAPH = None


def _get_graph():
    """Build graph lazily to keep frontend startup side-effect free."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_langgraph()
    return _GRAPH


def run_workflow(
    *,
    user_query: str,
    requested_outputs: List[str],
    export_requested: bool = False,
    export_formats: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Execute orchestration and return a deep-copied result for UI use.

    Frontend should never mutate orchestration internals directly.
    """
    safe_query = str(user_query).strip()
    safe_outputs = [str(item).strip() for item in requested_outputs if str(item).strip()]
    safe_export_formats = [
        str(item).strip().lower()
        for item in (export_formats or [])
        if str(item).strip()
    ]

    export_metadata = {
        "formats_requested": safe_export_formats if export_requested else [],
        "export_paths": {},
        "exported_at": None,
        "error_log": [],
    }

    initial_state = create_initial_state(
        user_query=safe_query,
        requested_outputs=safe_outputs,
        export_requested=bool(export_requested),
        export_metadata=export_metadata,
    )
    result = _get_graph().invoke(deepcopy(initial_state))
    if not isinstance(result, dict):
        return {}
    return deepcopy(result)
