"""Compatibility adapter for agent-facing image generation calls."""

from __future__ import annotations

import os
from typing import Any, Dict

from contentblitz.config import MODEL_FALLBACKS
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
    provider_primary = str(
        os.getenv("CONTENTBLITZ_IMAGE_PROVIDER")
        or MODEL_FALLBACKS.get("primary_image_provider", "stability_ai")
    ).strip()
    provider_fallback = str(
        os.getenv("CONTENTBLITZ_IMAGE_PROVIDER_FALLBACK")
        or MODEL_FALLBACKS.get("fallback_image_provider", "fal_ai")
    ).strip()
    primary_model = str(
        os.getenv("CONTENTBLITZ_IMAGE_MODEL_PRIMARY")
        or MODEL_FALLBACKS.get("primary_image_model", "stable-image-core")
    ).strip()

    result = _core_generate_image(
        prompt=prompt,
        model=primary_model,
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
        "provider_primary": provider_primary,
        "provider_fallback": provider_fallback,
        "provider_used": result.provider,
        "model_used": result.model,
        "provider_call_count": int(result.provider_call_count or 0),
        "provider_call_count_by_provider": dict(result.provider_call_count_by_provider),
        "provider_latency_by_provider_ms": dict(
            result.provider_latency_by_provider_ms
        ),
        "image_provider_attempts": list(result.provider_attempts),
        "primary_provider": result.primary_provider or provider_primary,
        "fallback_provider": result.fallback_provider or provider_fallback,
        "fallback_provider_attempted": bool(result.fallback_provider_attempted),
        "fallback_provider_used": bool(result.fallback_provider_used),
        "images": images,
        "used_external_api": not result.degraded,
        "degraded": result.degraded,
        "error": result.error,
    }
