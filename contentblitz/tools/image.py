"""Compatibility adapter for agent-facing image generation calls."""

from __future__ import annotations

from typing import Any, Dict

from contentblitz.tools.generate_image import generate_image as _core_generate_image


def _is_http_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def generate_image(prompt: str, style: str = "default") -> Dict[str, Any]:
    """
    Legacy dict contract used by existing agents.

    This adapter delegates to the typed `contentblitz.tools.generate_image`
    implementation and converts to the historical payload shape.
    Live-call gating (`CONTENTBLITZ_ENABLE_LIVE_CALLS`) is enforced in
    the delegated core tool.
    """
    result = _core_generate_image(
        prompt=prompt,
        model="dall-e-3",
        size="1024x1024",
        quality=None if style == "default" else None,
    )

    images = []
    if not result.degraded:
        image_item: Dict[str, Any] = {}
        if isinstance(result.image_url, str) and result.image_url.strip():
            cleaned_ref = result.image_url.strip()
            if _is_http_url(cleaned_ref):
                image_item["url"] = cleaned_ref
        if isinstance(result.local_path, str) and result.local_path.strip():
            image_item["local_path"] = result.local_path.strip()
        if isinstance(result.image_id, str) and result.image_id.strip():
            image_item["id"] = result.image_id.strip()
        image_item["renderable"] = bool(
            image_item.get("url") or image_item.get("local_path")
        )
        if result.revised_prompt:
            image_item["revised_prompt"] = result.revised_prompt
        if image_item:
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
