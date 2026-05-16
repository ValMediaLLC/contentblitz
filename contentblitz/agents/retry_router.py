"""Retry router node implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping

from contentblitz.config import RETRY_POLICY
from contentblitz.core.cost_controls import normalize_cost_controls, retry_cap_reached

_RETRY_TYPE_ORDER = ("blog", "linkedin", "image")
_AGENT_KEY_BY_TYPE = {
    "blog": "blog_writer",
    "linkedin": "linkedin_writer",
    "image": "image_agent",
}
_DEFAULT_MAX_TOTAL_RETRIES = 3


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _needs_retry(score: Mapping[str, Any]) -> bool:
    status = str(score.get("validation_status", "")).strip().lower()
    return status == "retry_needed"


def _determine_retry_types(state: Dict[str, Any]) -> List[str]:
    quality_scores = _safe_dict(state.get("quality_scores", {}))
    candidates: List[str] = []
    for content_type in _RETRY_TYPE_ORDER:
        score = _safe_dict(quality_scores.get(content_type, {}))
        if score and _needs_retry(score):
            candidates.append(content_type)
    return candidates


def _build_retry_feedback(content_type: str, score: Mapping[str, Any]) -> str:
    composite = score.get("composite")
    if isinstance(composite, (int, float)):
        composite_text = f"{float(composite):.2f}"
    else:
        composite_text = "unknown"
    return (
        f"Retry needed for {content_type}: validation_status=retry_needed, "
        f"composite={composite_text}. Improve clarity, structure, and usefulness."
    )


def retry_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute retry dispatch updates before retry routing executes."""
    retry_counts = deepcopy(_safe_dict(state.get("retry_counts", {})))
    retry_feedback = deepcopy(_safe_dict(state.get("retry_feedback", {})))
    cost_controls = normalize_cost_controls(_safe_dict(state.get("cost_controls", {})))
    quality_scores = _safe_dict(state.get("quality_scores", {}))

    retry_types = _determine_retry_types(state)
    if not retry_types:
        return {}

    if bool(cost_controls.get("budget_exceeded", False)) or retry_cap_reached(
        cost_controls
    ):
        return {
            "retry_requested": False,
            "retry_target": "",
            "retry_targets": [],
            "_retry_counts_incremented": False,
            "cost_controls": cost_controls,
        }

    total_retries_used = int(cost_controls.get("total_retries_used_this_session", 0))
    max_total_retries = int(
        cost_controls.get("max_total_retries_per_session", _DEFAULT_MAX_TOTAL_RETRIES)
    )
    remaining_session_retries = max(0, max_total_retries - total_retries_used)

    dispatch_types: List[str] = []
    for content_type in retry_types:
        if remaining_session_retries <= 0:
            break
        agent_key = _AGENT_KEY_BY_TYPE[content_type]
        used_for_agent = int(retry_counts.get(agent_key, 0))
        retry_limit_for_agent = int(RETRY_POLICY.get(agent_key, 0))
        if used_for_agent >= retry_limit_for_agent:
            continue
        retry_counts[agent_key] = used_for_agent + 1
        dispatch_types.append(content_type)
        remaining_session_retries -= 1

    if not dispatch_types:
        return {
            "retry_requested": False,
            "retry_target": "",
            "retry_targets": [],
            "_retry_counts_incremented": False,
        }

    for content_type in dispatch_types:
        if content_type not in {"blog", "linkedin"}:
            continue
        existing = retry_feedback.get(content_type)
        feedback_entries = _safe_list(existing)
        score = _safe_dict(quality_scores.get(content_type, {}))
        feedback_entries.append(_build_retry_feedback(content_type, score))
        retry_feedback[content_type] = feedback_entries

    cost_controls["total_retries_used_this_session"] = total_retries_used + len(
        dispatch_types
    )

    return {
        "retry_counts": retry_counts,
        "retry_feedback": retry_feedback,
        "cost_controls": cost_controls,
        "retry_requested": True,
        "retry_target": dispatch_types[0],
        "retry_targets": dispatch_types,
        "_retry_counts_incremented": True,
    }
