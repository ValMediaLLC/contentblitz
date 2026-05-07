"""Quality validator node implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _to_unit_score(raw: Any) -> float:
    if not isinstance(raw, (int, float)):
        raise ValueError("Composite score must be numeric.")
    score = float(raw)
    # Accept either 0..1 or 0..100 style scores.
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def _extract_composite(result: Mapping[str, Any]) -> float:
    for key in ("composite", "composite_score", "score", "quality_score"):
        if key in result:
            return _to_unit_score(result.get(key))
    raise ValueError("No composite score found in validation result.")


def _status_for_score(composite: float) -> Tuple[str, bool]:
    if composite >= 0.75:
        return "passed", True
    if composite >= 0.50:
        return "retry_needed", False
    return "failed", False


def validate_content(content_type: str, draft_body: str, context: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """
    Deterministic local scoring stub.

    This function is intentionally mockable in tests and does not call external APIs.
    """
    text = str(draft_body or "").strip()
    if not text:
        raise ValueError(f"Missing draft body for {content_type}.")

    words = [token for token in text.split() if token.strip()]
    word_count = len(words)

    # Deterministic heuristic tuned so first-pass placeholder drafts pass by default.
    if word_count >= 20:
        composite = 0.80
    elif word_count >= 8:
        composite = 0.66
    else:
        composite = 0.45

    return {
        "composite": composite,
        "metrics": {
            "word_count": word_count,
        },
    }


def quality_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Score blog/linkedin drafts and update quality-tracking state."""
    requested_outputs = {
        str(item).strip().lower()
        for item in _safe_list(state.get("requested_outputs", []))
        if str(item).strip()
    }
    content_drafts = _safe_dict(state.get("content_drafts", {}))
    draft_status = _safe_dict(state.get("draft_status", {}))

    quality_scores = deepcopy(_safe_dict(state.get("quality_scores", {})))
    best_drafts = deepcopy(_safe_dict(state.get("best_drafts", {})))
    attempt_history = deepcopy(_safe_dict(state.get("attempt_history", {})))
    errors = deepcopy(_safe_list(state.get("errors", [])))

    for key in ("blog", "linkedin", "image"):
        history = attempt_history.get(key, [])
        if not isinstance(history, list):
            history = []
        attempt_history[key] = history

    retry_candidates: List[str] = []

    for content_type in ("blog", "linkedin"):
        draft = _safe_dict(content_drafts.get(content_type, {}))
        body = str(draft.get("body", "")).strip()
        version = int(draft.get("version", 0))

        should_validate = (
            content_type in requested_outputs
            or bool(body)
            or str(draft_status.get(content_type, "")).strip().lower() == "complete"
            or str(state.get("retry_target", "")).strip().lower() == content_type
        )
        if not should_validate:
            continue

        if not body:
            errors.append(
                {
                    "agent": "quality_validator",
                    "type": "missing_draft",
                    "message": f"No draft body available for {content_type}.",
                    "recoverable": True,
                }
            )
            continue

        try:
            raw_result = validate_content(
                content_type=content_type,
                draft_body=body,
                context=state,
            )
            result = _safe_dict(raw_result)
            composite = round(_extract_composite(result), 2)
            validation_status, passed = _status_for_score(composite)
        except Exception as exc:
            composite = 0.60
            passed = False
            validation_status = "unverified"
            result = {
                "error": str(exc),
            }

        quality_scores[content_type] = {
            "composite": composite,
            "passed": passed,
            "validation_status": validation_status,
        }

        attempt_history[content_type].append(
            {
                "version": version,
                "composite": composite,
                "passed": passed,
                "validation_status": validation_status,
            }
        )

        previous_best = _safe_dict(best_drafts.get(content_type, {}))
        previous_composite = previous_best.get("composite")
        previous_numeric = (
            float(previous_composite)
            if isinstance(previous_composite, (int, float))
            else float("-inf")
        )
        if composite > previous_numeric:
            best_drafts[content_type] = {
                "version": version,
                "body": body,
                "composite": composite,
                "validation_status": validation_status,
            }

        if validation_status == "retry_needed":
            retry_candidates.append(content_type)

    incoming_retry_requested = bool(state.get("retry_requested", False))
    incoming_retry_target = str(state.get("retry_target", "")).strip().lower()

    if incoming_retry_requested and incoming_retry_target in {"blog", "linkedin"}:
        target = incoming_retry_target
        target_score = _safe_dict(quality_scores.get(target, {}))
        if target_score:
            target_score["validation_status"] = "retry_needed"
            target_score["passed"] = False
            quality_scores[target] = target_score
        retry_requested = True
        retry_target = target
    elif retry_candidates:
        retry_requested = True
        retry_target = retry_candidates[0]
    else:
        retry_requested = False
        retry_target = ""

    return {
        "quality_scores": quality_scores,
        "best_drafts": best_drafts,
        "attempt_history": attempt_history,
        "errors": errors,
        "retry_requested": retry_requested,
        "retry_target": retry_target,
    }
