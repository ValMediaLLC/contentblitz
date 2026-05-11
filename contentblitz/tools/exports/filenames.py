"""Filename and path safety helpers for export artifacts."""

from __future__ import annotations

import os
import re
from hashlib import sha256
from pathlib import Path

DEFAULT_EXPORT_DIR = "exports"
EXPORT_DIR_ENV_VAR = "CONTENTBLITZ_EXPORT_DIR"

_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _sanitize_filename_token(value: str) -> str:
    candidate = _SAFE_CHARS_RE.sub("_", _safe_text(value))
    candidate = candidate.strip("._-")
    return candidate or "content"


def resolve_export_dir(base_dir: str | Path | None = None) -> Path:
    if base_dir is not None:
        return Path(base_dir).expanduser()
    env_value = _safe_text(os.getenv(EXPORT_DIR_ENV_VAR, ""))
    if env_value:
        return Path(env_value).expanduser()
    return Path(DEFAULT_EXPORT_DIR)


def make_markdown_filename(seed_text: str) -> str:
    """Create deterministic markdown filename from a safe hash."""
    digest = sha256(_safe_text(seed_text).encode("utf-8")).hexdigest()[:12]
    return f"content_{digest}.md"


def resolve_markdown_export_path(
    seed_text: str,
    *,
    export_dir: str | Path | None = None,
) -> Path:
    """
    Resolve a markdown file path inside the configured export directory.

    This helper prevents path traversal by validating the resolved parent path.
    """
    target_dir = resolve_export_dir(export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = make_markdown_filename(seed_text)
    filename = _sanitize_filename_token(filename)
    if not filename.lower().endswith(".md"):
        filename = f"{filename}.md"

    candidate = (target_dir / filename).resolve()
    resolved_dir = target_dir.resolve()
    if candidate.parent != resolved_dir:
        raise ValueError("Unsafe export path outside configured export directory.")
    return candidate

