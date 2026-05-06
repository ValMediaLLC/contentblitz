"""Quality validator node scaffold."""

from __future__ import annotations

from typing import Any, Dict


def quality_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1 placeholder:
    when retry is requested, emit a deterministic validation_status signal
    used by routing tests.
    """
    if not bool(state.get("retry_requested", False)):
        return {}

    target = str(state.get("retry_target", "")).strip().lower()
    if target not in {"blog", "linkedin", "image"}:
        target = "blog"

    existing_scores = state.get("quality_scores", {})
    already_marked = (
        isinstance(existing_scores, dict)
        and isinstance(existing_scores.get(target), dict)
        and existing_scores[target].get("validation_status") == "retry_needed"
    )

    if already_marked:
        return {"retry_requested": False}

    return {
        "quality_scores": {
            target: {
                "validation_status": "retry_needed",
            }
        },
        "retry_requested": True,
        "retry_target": target,
    }
