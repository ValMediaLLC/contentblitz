"""OpenAI-backed image generation tool with safe fallback behavior."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from contentblitz.config import INJECTION_GUARD, live_provider_calls_enabled
from contentblitz.core.observability import safe_tool_metadata, start_tool_span
from contentblitz.tools.exports.filenames import resolve_export_dir

_PROVIDER = "openai"
_PRIMARY_MODEL = "dall-e-3"
_FALLBACK_MODEL = "dall-e-2"
_MODERN_IMAGE_MODEL = "gpt-image-1"
_DEFAULT_SIZE = "1024x1024"


@dataclass(frozen=True)
class GenerateImageResult:
    """Normalized image-generation result."""

    provider: str
    model: str
    prompt: str
    image_url: Optional[str]
    local_path: Optional[str]
    image_id: Optional[str]
    revised_prompt: Optional[str]
    renderable: bool
    degraded: bool
    error: Optional[Dict[str, Any]]


def _build_openai_client(api_key: str) -> OpenAI:
    """Factory wrapper to allow deterministic monkeypatching in tests."""
    return OpenAI(api_key=api_key)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_url_or_ref(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        return candidate
    return None


def _safe_image_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if lowered.startswith("data:image/") or "base64" in lowered:
        return None
    return candidate


def _safe_bytes_from_base64(value: Any) -> Optional[bytes]:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token:
        return None
    try:
        decoded = base64.b64decode(token, validate=True)
    except Exception:
        return None
    return decoded if decoded else None


def _safe_image_bytes(value: Any) -> Optional[bytes]:
    if isinstance(value, bytes):
        return value if value else None
    if isinstance(value, bytearray):
        blob = bytes(value)
        return blob if blob else None
    return None


def _extension_from_format(value: Any) -> str:
    token = _safe_text(value).strip().lower()
    if "/" in token:
        token = token.split("/")[-1].strip()
    if token in {"jpg", "jpeg"}:
        return "jpg"
    if token in {"webp"}:
        return "webp"
    if token in {"gif"}:
        return "gif"
    return "png"


def _write_image_bytes_to_local_path(
    *,
    image_bytes: bytes,
    prompt: str,
    model: str,
    extension: str,
) -> Optional[str]:
    if not image_bytes:
        return None

    export_dir = resolve_export_dir().resolve()
    image_dir = (export_dir / "images").resolve()
    image_dir.mkdir(parents=True, exist_ok=True)

    digest_seed = f"{prompt}|{model}|{hashlib.sha256(image_bytes).hexdigest()}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(digest_seed).hexdigest()[:24]
    safe_ext = _extension_from_format(extension)
    local_file = (image_dir / f"image_{digest}.{safe_ext}").resolve()
    if local_file.parent != image_dir:
        return None
    local_file.write_bytes(image_bytes)

    try:
        return local_file.relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return local_file.as_posix()


def _normalize_provider_error(exc: Exception) -> Dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    status_value = status_code if isinstance(status_code, int) else None

    if isinstance(exc, AuthenticationError):
        return {
            "code": "authentication_error",
            "message": "Authentication with the image provider failed.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": False,
        }
    if isinstance(exc, RateLimitError):
        return {
            "code": "rate_limited",
            "message": "The image provider rate limit was reached.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    if isinstance(exc, APIConnectionError):
        return {
            "code": "provider_unavailable",
            "message": "The image provider is temporarily unavailable.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    if isinstance(exc, BadRequestError):
        return {
            "code": "bad_request",
            "message": "The image provider rejected the request format.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": False,
        }
    if isinstance(exc, APIError):
        return {
            "code": "provider_error",
            "message": "The image provider returned an internal error.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    return {
        "code": "provider_error",
        "message": "The image provider request failed.",
        "provider": _PROVIDER,
        "status_code": status_value,
        "recoverable": True,
    }


def _should_try_modern_image_model(
    exc: Exception,
    *,
    normalized_error: Mapping[str, Any] | None,
    models_already_scheduled: list[str],
) -> bool:
    if _MODERN_IMAGE_MODEL in models_already_scheduled:
        return False
    code = _safe_text((normalized_error or {}).get("code")).lower()
    if code != "bad_request":
        return False
    message = _safe_text(str(exc)).lower()
    return "model" in message and "does not exist" in message


def _rejected_prompt_error(reason: str) -> Dict[str, Any]:
    return {
        "code": "prompt_rejected",
        "message": "Prompt failed safety checks.",
        "provider": _PROVIDER,
        "recoverable": False,
        "reason": reason,
    }


def _sanitize_prompt(prompt: str) -> str:
    if not INJECTION_GUARD.get("sanitize_user_input", False):
        return prompt
    return prompt.replace("\x00", " ").strip()


def _validate_prompt(prompt: str) -> Optional[Dict[str, Any]]:
    normalized = prompt.lower()
    max_len = INJECTION_GUARD.get("max_input_length", 0)
    if isinstance(max_len, int) and max_len > 0 and len(prompt) > max_len:
        return _rejected_prompt_error("max_length_exceeded")

    blocked_patterns = INJECTION_GUARD.get("blocked_patterns", [])
    if isinstance(blocked_patterns, list):
        for pattern in blocked_patterns:
            token = str(pattern).strip().lower()
            if token and token in normalized:
                return _rejected_prompt_error("blocked_pattern")
    return None


def _degraded_result(
    *,
    prompt: str,
    model: str,
    error: Dict[str, Any],
) -> GenerateImageResult:
    return GenerateImageResult(
        provider=_PROVIDER,
        model=model,
        prompt=prompt,
        image_url=None,
        local_path=None,
        image_id=None,
        revised_prompt=None,
        renderable=False,
        degraded=True,
        error=error,
    )


def _model_span_name(model: str) -> str:
    normalized = _safe_text(model).lower()
    if normalized == _PRIMARY_MODEL:
        return "dall_e_3"
    if normalized == _FALLBACK_MODEL:
        return "dall_e_2_fallback"
    if normalized == _MODERN_IMAGE_MODEL:
        return "gpt_image_1_fallback"
    return "image_provider_attempt"


def _finish_tool_span(
    *,
    span: Any,
    started_at: float,
    span_name: str,
    requested_model: str,
    fallback_used: bool,
    fallback_provider: str = "",
    fallback_model: str = "",
    fallback_reason: str = "",
    retry_attempt: int = 0,
    retry_exhausted: bool = False,
    result: GenerateImageResult | None = None,
    error: BaseException | None = None,
) -> None:
    duration_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    metadata: Dict[str, Any] = {
        "tool_name": span_name,
        "provider": _PROVIDER,
        "model": requested_model,
        "fallback_used": fallback_used,
        "fallback_provider": _safe_text(fallback_provider),
        "fallback_model": _safe_text(fallback_model),
        "fallback_reason": _safe_text(fallback_reason),
        "retry_attempt": max(0, int(retry_attempt)),
        "retry_exhausted": bool(retry_exhausted),
        "duration_ms": duration_ms,
    }
    outputs: Dict[str, Any] = {}
    if result is not None:
        has_image = bool(result.image_url or result.local_path)
        metadata.update(
            {
                "provider": result.provider,
                "model": result.model,
                "final_model": result.model,
                "degraded": result.degraded,
                "image_url_present": has_image,
                "result_count": 1 if has_image else 0,
                "image_output_count": 1 if has_image else 0,
            }
        )
        if fallback_used:
            metadata["fallback_provider"] = _PROVIDER
            metadata["fallback_model"] = result.model
            if not metadata["fallback_reason"]:
                metadata["fallback_reason"] = "model_fallback"
        if result.degraded and isinstance(result.error, Mapping):
            code = _safe_text(result.error.get("code"))
            if code:
                metadata["fallback_reason"] = code
        outputs = {
            "provider": result.provider,
            "model": result.model,
            "degraded": result.degraded,
            "renderable": result.renderable,
            "retry_exhausted": bool(retry_exhausted),
        }
    elif error is not None:
        metadata.update(
            {
                "degraded": True,
                "image_url_present": False,
                "fallback_reason": metadata.get("fallback_reason")
                or "tool_exception",
            }
        )
    span.finish(
        metadata=safe_tool_metadata(metadata),
        outputs=outputs,
        error=error,
    )


def _extract_data_item(response: Any) -> Optional[Mapping[str, Any]]:
    data = getattr(response, "data", None)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, Mapping):
            return first
        # openai SDK objects may expose attrs and model_dump
        model_dump = getattr(first, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, Mapping):
                return dumped
        if hasattr(first, "__dict__"):
            raw = vars(first)
            if isinstance(raw, Mapping):
                return raw
    return None


def _call_provider(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    size: str,
    quality: str | None,
) -> GenerateImageResult:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    if quality is not None and str(quality).strip():
        # DALL-E 2 does not support variable quality options beyond standard.
        if model != _FALLBACK_MODEL:
            payload["quality"] = str(quality).strip()

    response = client.images.generate(**payload)
    first_item = _extract_data_item(response)
    if first_item is None:
        return _degraded_result(
            prompt=prompt,
            model=model,
            error={
                "code": "provider_payload_unusable",
                "message": "Image provider returned no usable image payload.",
                "provider": _PROVIDER,
                "recoverable": True,
            },
        )

    image_url = _safe_url_or_ref(first_item.get("url"))
    local_path: Optional[str] = None
    image_id = _safe_image_id(first_item.get("file_id") or first_item.get("id"))

    image_bytes = _safe_image_bytes(first_item.get("image_bytes"))
    if image_bytes is None:
        image_bytes = _safe_image_bytes(first_item.get("bytes"))
    if image_bytes is None:
        image_bytes = _safe_bytes_from_base64(first_item.get("b64_json"))

    if image_url is None and image_bytes is not None:
        local_path = _write_image_bytes_to_local_path(
            image_bytes=image_bytes,
            prompt=prompt,
            model=model,
            extension=_safe_text(
                first_item.get("output_format") or first_item.get("mime_type")
            ),
        )

    revised_prompt = _safe_text(first_item.get("revised_prompt")) or None

    renderable = bool(image_url or local_path)
    if not renderable and image_id:
        return GenerateImageResult(
            provider=_PROVIDER,
            model=model,
            prompt=prompt,
            image_url=None,
            local_path=None,
            image_id=image_id,
            revised_prompt=revised_prompt,
            renderable=False,
            degraded=False,
            error=None,
        )

    if not renderable:
        return _degraded_result(
            prompt=prompt,
            model=model,
            error={
                "code": "provider_payload_unusable",
                "message": "Image provider returned no renderable image artifact.",
                "provider": _PROVIDER,
                "recoverable": True,
            },
        )

    return GenerateImageResult(
        provider=_PROVIDER,
        model=model,
        prompt=prompt,
        image_url=image_url,
        local_path=local_path,
        image_id=image_id,
        revised_prompt=revised_prompt,
        renderable=True,
        degraded=False,
        error=None,
    )


def generate_image(
    prompt: str,
    *,
    model: str = _PRIMARY_MODEL,
    size: str = _DEFAULT_SIZE,
    quality: str | None = None,
) -> GenerateImageResult:
    """Generate an image with DALL-E 3 primary and DALL-E 2 fallback."""
    raw_prompt = str(prompt or "")
    safe_prompt = _sanitize_prompt(raw_prompt)
    chosen_model = _safe_text(model) or _PRIMARY_MODEL
    started_at = time.perf_counter()
    tool_span = start_tool_span(
        "generate_image",
        metadata={
            "provider": _PROVIDER,
            "model": chosen_model,
        },
        inputs={"tool_name": "generate_image", "model": chosen_model},
    )

    attempt_counter = 0

    def _finalize(
        result: GenerateImageResult,
        *,
        retry_attempt: int = 0,
        retry_exhausted: bool = False,
    ) -> GenerateImageResult:
        fallback_used = result.model != chosen_model
        _finish_tool_span(
            span=tool_span,
            started_at=started_at,
            span_name="generate_image",
            requested_model=chosen_model,
            fallback_used=fallback_used,
            fallback_provider=_PROVIDER if fallback_used else "",
            fallback_model=result.model if fallback_used else "",
            fallback_reason="model_fallback" if fallback_used else "",
            retry_attempt=retry_attempt,
            retry_exhausted=retry_exhausted,
            result=result,
        )
        return result

    try:
        if not safe_prompt:
            return _finalize(
                _degraded_result(
                    prompt=safe_prompt,
                    model=chosen_model,
                    error=_rejected_prompt_error("empty_prompt"),
                )
            )

        blocked = _validate_prompt(safe_prompt)
        if blocked is not None:
            return _finalize(
                _degraded_result(
                    prompt=safe_prompt,
                    model=chosen_model,
                    error=blocked,
                )
            )

        if not live_provider_calls_enabled():
            return _finalize(
                _degraded_result(
                    prompt=safe_prompt,
                    model=chosen_model,
                    error={
                        "code": "live_calls_disabled",
                        "message": (
                            "Live provider calls are disabled by "
                            "CONTENTBLITZ_ENABLE_LIVE_CALLS."
                        ),
                        "provider": _PROVIDER,
                        "recoverable": False,
                    },
                )
            )

        api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            return _finalize(
                _degraded_result(
                    prompt=safe_prompt,
                    model=chosen_model,
                    error={
                        "code": "configuration_error",
                        "message": "OPENAI_API_KEY is not configured.",
                        "provider": _PROVIDER,
                        "recoverable": False,
                    },
                )
            )

        client = _build_openai_client(api_key=api_key)
        models_to_try = [chosen_model]
        if _FALLBACK_MODEL not in models_to_try:
            models_to_try.append(_FALLBACK_MODEL)

        last_error: Optional[Dict[str, Any]] = None
        index = 0
        while index < len(models_to_try):
            model_name = models_to_try[index]
            attempt_counter += 1
            child_name = _model_span_name(model_name)
            child_started_at = time.perf_counter()
            child_span = start_tool_span(
                child_name,
                metadata={
                    "provider": _PROVIDER,
                    "model": model_name,
                    "fallback_used": model_name != chosen_model,
                },
                inputs={"tool_name": child_name, "model": model_name},
            )
            try:
                result = _call_provider(
                    client,
                    model=model_name,
                    prompt=safe_prompt,
                    size=str(size or _DEFAULT_SIZE),
                    quality=quality,
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - deterministic via mocks in tests
                normalized = _normalize_provider_error(exc)
                last_error = normalized
                _finish_tool_span(
                    span=child_span,
                    started_at=child_started_at,
                    span_name=child_name,
                    requested_model=model_name,
                    fallback_used=model_name != chosen_model,
                    fallback_provider=_PROVIDER if model_name != chosen_model else "",
                    fallback_model=model_name if model_name != chosen_model else "",
                    fallback_reason="model_attempt_failed",
                    retry_attempt=attempt_counter,
                    retry_exhausted=False,
                    error=exc,
                )
                if _should_try_modern_image_model(
                    exc,
                    normalized_error=normalized,
                    models_already_scheduled=models_to_try,
                ):
                    models_to_try.append(_MODERN_IMAGE_MODEL)
                index += 1
                continue

            _finish_tool_span(
                span=child_span,
                started_at=child_started_at,
                span_name=child_name,
                requested_model=model_name,
                fallback_used=model_name != chosen_model,
                fallback_provider=_PROVIDER if model_name != chosen_model else "",
                fallback_model=model_name if model_name != chosen_model else "",
                fallback_reason="",
                retry_attempt=attempt_counter,
                retry_exhausted=False,
                result=result,
            )
            if result.degraded:
                last_error = result.error
                index += 1
                continue
            return _finalize(
                result,
                retry_attempt=attempt_counter,
                retry_exhausted=False,
            )

        return _finalize(
            _degraded_result(
                prompt=safe_prompt,
                model=models_to_try[-1],
                error={
                    "code": "provider_failure",
                    "message": (
                        "Image generation failed after primary and fallback models."
                    ),
                    "provider": _PROVIDER,
                    "recoverable": True,
                    "models_attempted": models_to_try,
                    "last_error": last_error,
                },
            ),
            retry_attempt=attempt_counter,
            retry_exhausted=True,
        )
    except Exception as error:
        _finish_tool_span(
            span=tool_span,
            started_at=started_at,
            span_name="generate_image",
            requested_model=chosen_model,
            fallback_used=False,
            fallback_provider="",
            fallback_model="",
            fallback_reason="tool_exception",
            retry_attempt=attempt_counter,
            retry_exhausted=False,
            error=error,
        )
        raise


__all__ = ["GenerateImageResult", "generate_image"]
