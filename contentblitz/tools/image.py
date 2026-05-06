"""Image generation tool interface scaffold."""

from __future__ import annotations

from typing import Any, Dict


def generate_image(prompt: str, style: str = "default") -> Dict[str, Any]:
    """Return a deterministic placeholder without external calls."""
    return {
        "prompt": prompt,
        "style": style,
        "provider_primary": "dall-e-3",
        "provider_fallback": "dall-e-2",
        "images": [],
        "used_external_api": False,
    }

