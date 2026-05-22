"""Image generation tool with provider registry and safe fallback behavior."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from contentblitz.config import (
    INJECTION_GUARD,
    MODEL_FALLBACKS,
    live_provider_calls_enabled,
)
from contentblitz.core.observability import safe_tool_metadata, start_tool_span
from contentblitz.tools.exports.filenames import resolve_export_dir

_PROVIDER_STABILITY = "stability_ai"
_PROVIDER_FAL = "fal_ai"
_SUPPORTED_PROVIDERS = {_PROVIDER_STABILITY, _PROVIDER_FAL}

_DEFAULT_PRIMARY_PROVIDER = str(
    MODEL_FALLBACKS.get("primary_image_provider", _PROVIDER_STABILITY)
).strip()
_DEFAULT_FALLBACK_PROVIDER = str(
    MODEL_FALLBACKS.get("fallback_image_provider", _PROVIDER_FAL)
).strip()

_DEFAULT_STABILITY_MODEL = str(
    MODEL_FALLBACKS.get("primary_image_model", "stable-image-core")
).strip()
_DEFAULT_FAL_MODEL = str(
    MODEL_FALLBACKS.get("fallback_image_model", "fal-ai/fast-sdxl")
).strip()

_DEFAULT_SIZE = "1024x1024"
_STABILITY_CORE_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/core"
_FAL_RUN_ENDPOINT = "https://fal.run"
_SAFE_PROVIDER_ERROR_CODES = {
    "quota_exceeded",
    "authentication_failed",
    "rate_limited",
    "provider_unavailable",
    "empty_provider_response",
    "unknown_provider_error",
    "configuration_error",
}
_PROVIDER_INPUT_ALIASES = {
    "stability": _PROVIDER_STABILITY,
    "stability_ai": _PROVIDER_STABILITY,
    "stabilityai": _PROVIDER_STABILITY,
    "fal": _PROVIDER_FAL,
    "fal_ai": _PROVIDER_FAL,
    "fal.ai": _PROVIDER_FAL,
    "falai": _PROVIDER_FAL,
}
_STABILITY_ASPECT_BY_SIZE = {
    "1024x1024": "1:1",
    "1024x1536": "2:3",
    "1536x1024": "3:2",
    "1152x896": "9:7",
    "896x1152": "7:9",
    "1344x768": "16:9",
    "768x1344": "9:16",
}


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


@dataclass(frozen=True)
class _ProviderRequestError(Exception):
    status_code: int
    detail: str

    def __str__(self) -> str:
        return self.detail


class _StabilityHTTPClient:
    def __init__(self, *, api_key: str, endpoint: str | None = None) -> None:
        self.api_key = api_key
        self.endpoint = str(endpoint or _STABILITY_CORE_ENDPOINT).strip()

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        size: str,
        quality: str | None,
    ) -> Dict[str, Any]:
        _ = (model, quality)
        aspect_ratio = _STABILITY_ASPECT_BY_SIZE.get(size, "1:1")
        payload = urllib.parse.urlencode(
            {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "image/*",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content_type = str(response.headers.get("Content-Type", "")).lower()
                raw = response.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - live-only path
            body = exc.read().decode("utf-8", errors="ignore")
            raise _ProviderRequestError(
                status_code=int(exc.code),
                detail=body or "stability request failed",
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - live-only path
            raise _ProviderRequestError(
                status_code=0,
                detail=str(exc.reason),
            ) from exc

        if "application/json" in content_type:
            parsed = _parse_json_bytes(raw)
            image_base64 = _safe_text(parsed.get("image"))
            image_bytes = (
                _safe_bytes_from_base64(image_base64) if image_base64 else None
            )
            return {
                "image_bytes": image_bytes,
                "revised_prompt": None,
                "mime_type": "image/png",
            }

        return {
            "image_bytes": raw,
            "revised_prompt": None,
            "mime_type": content_type or "image/png",
        }


class _FalHTTPClient:
    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = str(base_url or _FAL_RUN_ENDPOINT).rstrip("/")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        size: str,
        quality: str | None,
    ) -> Dict[str, Any]:
        aspect_ratio = _STABILITY_ASPECT_BY_SIZE.get(size, "1:1")
        body = {
            "prompt": prompt,
            "image_size": aspect_ratio,
        }
        if quality:
            body["quality"] = str(quality).strip()
        payload = json.dumps(body).encode("utf-8")
        endpoint = f"{self.base_url}/{model.lstrip('/')}"
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - live-only path
            detail = exc.read().decode("utf-8", errors="ignore")
            raise _ProviderRequestError(
                status_code=int(exc.code),
                detail=detail or "fal request failed",
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - live-only path
            raise _ProviderRequestError(
                status_code=0,
                detail=str(exc.reason),
            ) from exc
        return _parse_json_bytes(raw)


def _build_stability_client(api_key: str) -> Any:
    """Factory wrapper to allow deterministic monkeypatching in tests."""
    return _StabilityHTTPClient(api_key=api_key)


def _build_fal_client(api_key: str) -> Any:
    """Factory wrapper to allow deterministic monkeypatching in tests."""
    return _FalHTTPClient(api_key=api_key)


def _build_openai_client(api_key: str) -> Any:
    """Backward-compatible test seam for historical monkeypatches."""
    return _build_stability_client(api_key)


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


def _parse_json_bytes(raw: bytes) -> Dict[str, Any]:
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _normalize_provider_name(
    value: str | None,
    *,
    default: str,
) -> str:
    token = _safe_text(value).lower()
    if not token:
        return default
    normalized = _PROVIDER_INPUT_ALIASES.get(token, token)
    if normalized in _SUPPORTED_PROVIDERS:
        return normalized
    return default


def _provider_default_model(provider: str) -> str:
    if provider == _PROVIDER_FAL:
        return _DEFAULT_FAL_MODEL
    return _DEFAULT_STABILITY_MODEL


def _provider_api_key_env_names(provider: str) -> tuple[str, ...]:
    if provider == _PROVIDER_FAL:
        return ("FAL_API_KEY", "FAL_KEY", "OPENAI_API_KEY")
    return ("STABILITY_API_KEY", "OPENAI_API_KEY")


def _read_provider_api_key(provider: str) -> Optional[str]:
    for env_name in _provider_api_key_env_names(provider):
        raw = os.getenv(env_name)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    return None


def _normalize_provider_error(exc: Exception, provider: str) -> Dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    status_value = status_code if isinstance(status_code, int) else None
    message_text = str(exc).lower()
    error_class = exc.__class__.__name__.strip().lower() or "unknown_error"

    if status_value in {401, 403} or "authentication" in error_class:
        return {
            "code": "authentication_failed",
            "message": "Image generation provider authentication failed.",
            "provider": provider,
            "status_code": status_value,
            "error_class": error_class,
            "recoverable": False,
        }
    if status_value == 429 or "rate" in message_text:
        is_quota_error = "quota" in message_text or "billing" in message_text
        return {
            "code": "quota_exceeded" if is_quota_error else "rate_limited",
            "message": (
                "Image generation provider is unavailable or quota-limited."
                if is_quota_error
                else "Image generation provider is rate-limited."
            ),
            "provider": provider,
            "status_code": status_value,
            "error_class": error_class,
            "recoverable": True,
        }
    if status_value and status_value >= 500:
        return {
            "code": "provider_unavailable",
            "message": "Image generation provider is temporarily unavailable.",
            "provider": provider,
            "status_code": status_value,
            "error_class": error_class,
            "recoverable": True,
        }
    if isinstance(exc, _ProviderRequestError):
        return {
            "code": "provider_unavailable",
            "message": "Image generation provider is temporarily unavailable.",
            "provider": provider,
            "status_code": (
                status_value if status_value is not None else exc.status_code
            ),
            "error_class": error_class,
            "recoverable": True,
        }
    return {
        "code": "unknown_provider_error",
        "message": "Image generation provider request failed safely.",
        "provider": provider,
        "status_code": status_value,
        "error_class": error_class,
        "recoverable": True,
    }


def _rejected_prompt_error(reason: str, provider: str) -> Dict[str, Any]:
    return {
        "code": "prompt_rejected",
        "message": "Prompt failed safety checks.",
        "provider": provider,
        "recoverable": False,
        "reason": reason,
    }


def _sanitize_prompt(prompt: str) -> str:
    if not INJECTION_GUARD.get("sanitize_user_input", False):
        return prompt
    return prompt.replace("\x00", " ").strip()


def _validate_prompt(prompt: str, provider: str) -> Optional[Dict[str, Any]]:
    normalized = prompt.lower()
    max_len = INJECTION_GUARD.get("max_input_length", 0)
    if isinstance(max_len, int) and max_len > 0 and len(prompt) > max_len:
        return _rejected_prompt_error("max_length_exceeded", provider)

    blocked_patterns = INJECTION_GUARD.get("blocked_patterns", [])
    if isinstance(blocked_patterns, list):
        for pattern in blocked_patterns:
            token = str(pattern).strip().lower()
            if token and token in normalized:
                return _rejected_prompt_error("blocked_pattern", provider)
    return None


def _degraded_result(
    *,
    prompt: str,
    provider: str,
    model: str,
    error: Dict[str, Any],
) -> GenerateImageResult:
    return GenerateImageResult(
        provider=provider,
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


def _model_span_name(provider: str, *, fallback: bool) -> str:
    if provider == _PROVIDER_STABILITY:
        return "stability_ai_fallback" if fallback else "stability_ai"
    if provider == _PROVIDER_FAL:
        return "fal_ai_fallback" if fallback else "fal_ai"
    return "image_provider_attempt"


def _finish_tool_span(
    *,
    span: Any,
    started_at: float,
    span_name: str,
    requested_provider: str,
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
        "provider": requested_provider,
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
            metadata["fallback_provider"] = result.provider
            metadata["fallback_model"] = result.model
            if not metadata["fallback_reason"]:
                metadata["fallback_reason"] = "provider_or_model_fallback"
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
    if isinstance(response, Mapping):
        images = response.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, Mapping):
                return first
        for key in ("image", "output"):
            maybe = response.get(key)
            if isinstance(maybe, Mapping):
                return maybe
        return response

    data = getattr(response, "data", None)
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, Mapping):
            return first
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


def _result_from_payload(
    *,
    response: Any,
    provider: str,
    model: str,
    prompt: str,
) -> GenerateImageResult:
    first_item = _extract_data_item(response)
    if first_item is None:
        return _degraded_result(
            prompt=prompt,
            provider=provider,
            model=model,
            error={
                "code": "empty_provider_response",
                "message": "Image generation provider returned no usable output.",
                "provider": provider,
                "recoverable": True,
            },
        )

    image_url = _safe_url_or_ref(first_item.get("url") or first_item.get("image_url"))
    local_path = _safe_text(first_item.get("local_path")) or None
    image_id = _safe_image_id(first_item.get("file_id") or first_item.get("id"))
    image_bytes = _safe_image_bytes(first_item.get("image_bytes"))
    if image_bytes is None:
        image_bytes = _safe_image_bytes(first_item.get("bytes"))
    if image_bytes is None:
        image_bytes = _safe_bytes_from_base64(first_item.get("b64_json"))
    if image_bytes is None:
        image_bytes = _safe_bytes_from_base64(first_item.get("image"))

    if image_url is None and local_path is None and image_bytes is not None:
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
            provider=provider,
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
            provider=provider,
            model=model,
            error={
                "code": "empty_provider_response",
                "message": "Image generation provider returned no renderable output.",
                "provider": provider,
                "recoverable": True,
            },
        )

    return GenerateImageResult(
        provider=provider,
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


def _call_stability_provider(
    client: Any,
    *,
    provider: str,
    model: str,
    prompt: str,
    size: str,
    quality: str | None,
) -> GenerateImageResult:
    if hasattr(client, "generate"):
        response = client.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
        )
    elif hasattr(client, "images") and hasattr(client.images, "generate"):
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
        )
    else:
        raise RuntimeError("unsupported_stability_client")
    return _result_from_payload(
        response=response,
        provider=provider,
        model=model,
        prompt=prompt,
    )


def _call_fal_provider(
    client: Any,
    *,
    provider: str,
    model: str,
    prompt: str,
    size: str,
    quality: str | None,
) -> GenerateImageResult:
    if hasattr(client, "generate"):
        response = client.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
        )
    elif hasattr(client, "images") and hasattr(client.images, "generate"):
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
        )
    else:
        raise RuntimeError("unsupported_fal_client")
    return _result_from_payload(
        response=response,
        provider=provider,
        model=model,
        prompt=prompt,
    )


def _call_provider(
    *,
    client: Any,
    provider: str,
    model: str,
    prompt: str,
    size: str,
    quality: str | None,
) -> GenerateImageResult:
    if provider == _PROVIDER_FAL:
        return _call_fal_provider(
            client,
            provider=provider,
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
        )
    return _call_stability_provider(
        client,
        provider=provider,
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
    )


def _build_provider_client(provider: str, api_key: str) -> Any:
    if provider == _PROVIDER_FAL:
        return _build_fal_client(api_key)
    # Keep historical monkeypatch seam compatibility for tests that still patch
    # `_build_openai_client` while Stability is the primary image provider.
    return _build_openai_client(api_key)


def _build_provider_candidates(
    *,
    primary_provider: str,
    primary_model: str,
    fallback_provider: str,
    fallback_model: str,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for provider, model in (
        (primary_provider, primary_model),
        (fallback_provider, fallback_model),
    ):
        normalized_provider = _normalize_provider_name(
            provider,
            default=_PROVIDER_STABILITY,
        )
        normalized_model = _safe_text(model) or _provider_default_model(
            normalized_provider
        )
        candidate = (normalized_provider, normalized_model)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def generate_image(
    prompt: str,
    *,
    model: str = "",
    size: str = _DEFAULT_SIZE,
    quality: str | None = None,
) -> GenerateImageResult:
    """Generate an image with Stability AI primary and fal.ai fallback."""
    raw_prompt = str(prompt or "")
    safe_prompt = _sanitize_prompt(raw_prompt)
    primary_provider = _normalize_provider_name(
        os.getenv("CONTENTBLITZ_IMAGE_PROVIDER")
        or _DEFAULT_PRIMARY_PROVIDER,
        default=_PROVIDER_STABILITY,
    )
    fallback_provider = _normalize_provider_name(
        os.getenv("CONTENTBLITZ_IMAGE_PROVIDER_FALLBACK")
        or _DEFAULT_FALLBACK_PROVIDER,
        default=_PROVIDER_FAL,
    )
    primary_model = (
        _safe_text(model)
        or _safe_text(os.getenv("CONTENTBLITZ_IMAGE_MODEL_PRIMARY"))
        or _provider_default_model(primary_provider)
    )
    fallback_model = (
        _safe_text(os.getenv("CONTENTBLITZ_IMAGE_MODEL_FALLBACK"))
        or _provider_default_model(fallback_provider)
    )

    started_at = time.perf_counter()
    tool_span = start_tool_span(
        "generate_image",
        metadata={
            "provider": primary_provider,
            "model": primary_model,
        },
        inputs={
            "tool_name": "generate_image",
            "provider": primary_provider,
            "model": primary_model,
        },
    )
    attempt_counter = 0

    def _finalize(
        result: GenerateImageResult,
        *,
        retry_attempt: int = 0,
        retry_exhausted: bool = False,
    ) -> GenerateImageResult:
        fallback_used = (
            result.provider != primary_provider or result.model != primary_model
        )
        _finish_tool_span(
            span=tool_span,
            started_at=started_at,
            span_name="generate_image",
            requested_provider=primary_provider,
            requested_model=primary_model,
            fallback_used=fallback_used,
            fallback_provider=result.provider if fallback_used else "",
            fallback_model=result.model if fallback_used else "",
            fallback_reason="provider_or_model_fallback" if fallback_used else "",
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
                    provider=primary_provider,
                    model=primary_model,
                    error=_rejected_prompt_error("empty_prompt", primary_provider),
                )
            )

        blocked = _validate_prompt(safe_prompt, primary_provider)
        if blocked is not None:
            return _finalize(
                _degraded_result(
                    prompt=safe_prompt,
                    provider=primary_provider,
                    model=primary_model,
                    error=blocked,
                )
            )

        if not live_provider_calls_enabled():
            return _finalize(
                _degraded_result(
                    prompt=safe_prompt,
                    provider=primary_provider,
                    model=primary_model,
                    error={
                        "code": "live_calls_disabled",
                        "message": (
                            "Live provider calls are disabled by "
                            "CONTENTBLITZ_ENABLE_LIVE_CALLS."
                        ),
                        "provider": primary_provider,
                        "recoverable": False,
                    },
                )
            )

        candidates = _build_provider_candidates(
            primary_provider=primary_provider,
            primary_model=primary_model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )
        attempted_models: list[str] = []
        attempted_provider_models: list[str] = []
        provider_clients: dict[str, Any] = {}
        last_error: Optional[Dict[str, Any]] = None

        for index, (provider_name, model_name) in enumerate(candidates):
            attempt_counter += 1
            attempted_models.append(model_name)
            attempted_provider_models.append(f"{provider_name}:{model_name}")

            api_key = _read_provider_api_key(provider_name)
            if not api_key:
                last_error = {
                    "code": "configuration_error",
                    "message": "Image provider credentials are not configured.",
                    "provider": provider_name,
                    "recoverable": provider_name != primary_provider,
                }
                continue

            if provider_name not in provider_clients:
                try:
                    provider_clients[provider_name] = _build_provider_client(
                        provider_name,
                        api_key,
                    )
                except Exception as exc:
                    last_error = _normalize_provider_error(exc, provider_name)
                    continue

            child_started_at = time.perf_counter()
            child_span_name = _model_span_name(
                provider_name,
                fallback=index > 0,
            )
            child_span = start_tool_span(
                child_span_name,
                metadata={
                    "provider": provider_name,
                    "model": model_name,
                    "fallback_used": index > 0,
                },
                inputs={
                    "tool_name": child_span_name,
                    "provider": provider_name,
                    "model": model_name,
                },
            )
            try:
                result = _call_provider(
                    client=provider_clients[provider_name],
                    provider=provider_name,
                    model=model_name,
                    prompt=safe_prompt,
                    size=str(size or _DEFAULT_SIZE),
                    quality=quality,
                )
            except Exception as exc:  # pragma: no cover - deterministic via tests
                normalized = _normalize_provider_error(exc, provider_name)
                last_error = normalized
                _finish_tool_span(
                    span=child_span,
                    started_at=child_started_at,
                    span_name=child_span_name,
                    requested_provider=provider_name,
                    requested_model=model_name,
                    fallback_used=index > 0,
                    fallback_provider=provider_name if index > 0 else "",
                    fallback_model=model_name if index > 0 else "",
                    fallback_reason="provider_attempt_failed",
                    retry_attempt=attempt_counter,
                    retry_exhausted=False,
                    error=exc,
                )
                continue

            _finish_tool_span(
                span=child_span,
                started_at=child_started_at,
                span_name=child_span_name,
                requested_provider=provider_name,
                requested_model=model_name,
                fallback_used=index > 0,
                fallback_provider=provider_name if index > 0 else "",
                fallback_model=model_name if index > 0 else "",
                fallback_reason="",
                retry_attempt=attempt_counter,
                retry_exhausted=False,
                result=result,
            )
            if result.degraded:
                last_error = result.error
                continue
            return _finalize(
                result,
                retry_attempt=attempt_counter,
                retry_exhausted=False,
            )

        final_provider, final_model = candidates[-1] if candidates else (
            primary_provider,
            primary_model,
        )
        safe_error_code = (
            _safe_text((last_error or {}).get("code")).lower()
            if isinstance(last_error, Mapping)
            else ""
        )
        if safe_error_code not in _SAFE_PROVIDER_ERROR_CODES:
            safe_error_code = "unknown_provider_error"

        return _finalize(
            _degraded_result(
                prompt=safe_prompt,
                provider=final_provider,
                model=final_model,
                error={
                    "code": safe_error_code,
                    "message": (
                        "Image generation provider is unavailable or quota-limited."
                    ),
                    "provider": final_provider,
                    "recoverable": True,
                    "models_attempted": attempted_models,
                    "provider_models_attempted": attempted_provider_models,
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
            requested_provider=primary_provider,
            requested_model=primary_model,
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
