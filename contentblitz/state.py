"""State model for ContentBlitz Phase 1 scaffold."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any, Dict, List, Optional

from contentblitz.config import (
    RETRY_POLICY,
    build_cache_metadata_defaults,
    build_cost_controls_defaults,
)


def merge_content_drafts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer-compatible merge for parallel content_drafts updates."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            for nested_key, nested_value in value.items():
                if nested_value is None:
                    continue
                nested[nested_key] = nested_value
            merged[key] = nested
        elif isinstance(value, dict):
            merged[key] = dict(value)
        else:
            merged[key] = value
    return merged


def merge_draft_status(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer-compatible merge for parallel draft status updates."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if value is None:
            continue
        merged[key] = value
    return merged


@dataclass
class ContentBlitzState:
    session_id: str = ""
    user_id: str = ""
    user_query: str = ""

    intent: str = ""
    routing_decision: str = ""
    requested_outputs: List[str] = field(default_factory=list)

    conversation_history: List[Any] = field(default_factory=list)

    research_required: bool = False
    clarification_needed: bool = False
    clarification_message: Optional[str] = None

    research_data: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)

    content_brief: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "blog": {},
            "linkedin": {},
            "image": {},
        }
    )

    content_drafts: Annotated[Dict[str, Dict[str, Any]], merge_content_drafts] = field(
        default_factory=lambda: {
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        }
    )
    draft_status: Annotated[Dict[str, str], merge_draft_status] = field(
        default_factory=dict
    )

    best_drafts: Dict[str, Optional[Dict[str, Any]]] = field(
        default_factory=lambda: {
            "blog": None,
            "linkedin": None,
        }
    )

    attempt_history: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {
            "blog": [],
            "linkedin": [],
            "image": [],
        }
    )

    retry_feedback: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "blog": [],
            "linkedin": [],
        }
    )

    retry_counts: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in RETRY_POLICY}
    )

    quality_scores: Dict[str, Any] = field(default_factory=dict)

    image_prompts: List[str] = field(default_factory=list)
    image_outputs: List[Dict[str, Any]] = field(default_factory=list)

    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    final_response: str = ""
    assembled_outputs: Dict[str, Any] = field(default_factory=dict)
    export_outputs: Dict[str, Any] = field(default_factory=dict)
    workflow_status: str = ""

    export_requested: bool = False

    export_metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "formats_requested": [],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        }
    )

    cache_metadata: Dict[str, Any] = field(
        default_factory=build_cache_metadata_defaults
    )
    cost_controls: Dict[str, Any] = field(default_factory=build_cost_controls_defaults)


def create_initial_state(**overrides: Any) -> Dict[str, Any]:
    """Create a state dictionary with optional field overrides."""
    state = asdict(ContentBlitzState())
    if overrides:
        state.update(deepcopy(overrides))
    return state
