"""Retry router node scaffold."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def retry_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Increment session retry counter when retry routing is active.

    Routing target/ordering remains defined in core/workflow routing utilities.
    """
    if not bool(state.get("retry_requested", False)):
        return {}

    cost_controls = state.get("cost_controls", {})
    if not isinstance(cost_controls, dict):
        cost_controls = {}
    updated_cost_controls = deepcopy(cost_controls)
    updated_cost_controls["total_retries_used_this_session"] = (
        int(updated_cost_controls.get("total_retries_used_this_session", 0)) + 1
    )
    return {"cost_controls": updated_cost_controls}
