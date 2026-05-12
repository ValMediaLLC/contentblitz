"""Persistence models for Phase 3 local session storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class PersistedRunSummary:
    run_id: str
    session_id: str
    created_at: str
    updated_at: str
    user_query_preview: str
    requested_outputs: List[str] = field(default_factory=list)
    workflow_status: str = ""
    export_available: bool = False


@dataclass(frozen=True)
class PersistedRunRecord:
    run_id: str
    session_id: str
    created_at: str
    updated_at: str
    user_query: str
    requested_outputs: List[str]
    workflow_status: str
    routing_decision: str
    final_response: str
    content_drafts: Dict[str, Any] = field(default_factory=dict)
    partial_outputs: Dict[str, str] = field(default_factory=dict)
    partial_output_mode: str = "none"
    image_prompts: List[str] = field(default_factory=list)
    image_outputs: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: Dict[str, Any] = field(default_factory=dict)
    export_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    progress_events: List[Dict[str, Any]] = field(default_factory=list)
    status_messages: List[str] = field(default_factory=list)
    ui_selected_options: Dict[str, Any] = field(default_factory=dict)
    ui_node_statuses: Dict[str, str] = field(default_factory=dict)
    ui_workflow_status: str = ""
    prompt_injection_detected: bool = False
    prompt_injection_signals: List[str] = field(default_factory=list)
    sanitized_user_query: str = ""
