"""Normalization helpers for frontend workflow submission controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class WorkflowControls:
    include_blog: bool
    include_linkedin: bool
    include_research: bool
    include_image: bool
    export_enabled: bool
    export_formats: List[str]


def build_requested_outputs(controls: WorkflowControls) -> List[str]:
    """
    Build requested outputs in a deterministic order.

    This reflects UI preferences only. Orchestrator routing/classification remains
    authoritative for final execution behavior.
    """
    outputs: list[str] = []
    if controls.include_blog:
        outputs.append("blog")
    if controls.include_linkedin:
        outputs.append("linkedin")
    if controls.include_research:
        outputs.append("research")
    if controls.include_image:
        outputs.append("image")
    return outputs


def sanitize_export_formats(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values:
        normalized = str(raw).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned
