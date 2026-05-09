"""Compatibility adapter for agent-facing image generation calls."""

from __future__ import annotations

from typing import Any, Dict

from contentblitz.tools.generate_image import generate_image as _core_generate_image


def generate_image(prompt: str, style: str = "default") -> Dict[str, Any]:
    """
    Legacy dict contract used by existing agents.

    This adapter delegates to the typed `contentblitz.tools.generate_image`
    implementation and converts to the historical payload shape.
    """
    result = _core_generate_image(
        prompt=prompt,
        model="dall-e-3",
        size="1024x1024",
        quality=None if style == "default" else None,
    )

    images = []
    if not result.degraded and isinstance(result.image_url, str) and result.image_url.strip():
        image_item: Dict[str, Any] = {"url": result.image_url}
        if result.revised_prompt:
            image_item["revised_prompt"] = result.revised_prompt
        images.append(image_item)

    return {
        "prompt": prompt,
        "style": style,
        "provider_primary": "dall-e-3",
        "provider_fallback": "dall-e-2",
        "provider_used": result.model,
        "images": images,
        "used_external_api": not result.degraded,
        "degraded": result.degraded,
        "error": result.error,
    }
