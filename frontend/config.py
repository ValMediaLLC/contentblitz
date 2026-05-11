"""Shared frontend configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FrontendConfig:
    app_title: str = "ContentBlitz"
    app_icon: str = "CB"
    page_title_suffix: str = "Phase 3 UI Shell"
    default_query_placeholder: str = "Write a short LinkedIn post about AI content workflows."
    default_outputs: Tuple[str, ...] = ("blog", "linkedin")
    available_outputs: Tuple[str, ...] = ("research", "blog", "linkedin", "image")
    export_formats: Tuple[str, ...] = ("markdown", "html", "pdf", "docx")
    default_export_formats: Tuple[str, ...] = ("markdown",)
    logo_path: str = "frontend/assets/contentblitz_logo.svg"
    logo_icon_path: str = "frontend/assets/contentblitz_icon.svg"


FRONTEND_CONFIG = FrontendConfig()
