"""OpenAI-backed text generation tool with deterministic safety behavior."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from contentblitz.config import INJECTION_GUARD, RETRY_POLICY, live_provider_calls_enabled

_PROVIDER = "openai"
_PRIMARY_MODEL = "gpt-4o"
_FALLBACK_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class GenerateTextResult:
    """Normalized text-generation result returned by the provider tool."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    degraded: bool
    error: Optional[Dict[str, Any]]


def _build_openai_client(api_key: str) -> OpenAI:
    """Factory function to enable deterministic client mocking in tests."""
    return OpenAI(api_key=api_key)


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return default


def _extract_text_from_content_items(items: Iterable[Any]) -> str:
    fragments: list[str] = []
    for item in items:
        text_value = ""
        if isinstance(item, str):
            text_value = item
        elif isinstance(item, Mapping):
            text_value = str(item.get("text", "")).strip()
        else:
            candidate = getattr(item, "text", "")
            if isinstance(candidate, str):
                text_value = candidate.strip()

        if text_value:
            fragments.append(text_value)
    return "\n".join(fragments).strip()


def _extract_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return _extract_text_from_content_items(content)

    return ""


def _extract_usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0

    input_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
    output_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
    total_tokens = _safe_int(
        getattr(usage, "total_tokens", input_tokens + output_tokens),
        default=input_tokens + output_tokens,
    )
    return input_tokens, output_tokens, total_tokens


def _normalize_provider_error(exc: Exception) -> Dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    status_value = status_code if isinstance(status_code, int) else None

    if isinstance(exc, AuthenticationError):
        return {
            "code": "authentication_error",
            "message": "Authentication with the text provider failed.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": False,
        }
    if isinstance(exc, RateLimitError):
        return {
            "code": "rate_limited",
            "message": "The text provider rate limit was reached.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    if isinstance(exc, APIConnectionError):
        return {
            "code": "provider_unavailable",
            "message": "The text provider is temporarily unavailable.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    if isinstance(exc, BadRequestError):
        return {
            "code": "bad_request",
            "message": "The text provider rejected the request format.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": False,
        }
    if isinstance(exc, APIError):
        return {
            "code": "provider_error",
            "message": "The text provider returned an internal error.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }

    return {
        "code": "provider_error",
        "message": "The text provider request failed.",
        "provider": _PROVIDER,
        "status_code": status_value,
        "recoverable": True,
    }


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
    cleaned = prompt.replace("\x00", " ").strip()
    return cleaned


def _validate_prompt(prompt: str) -> Optional[Dict[str, Any]]:
    normalized_prompt = prompt.lower()
    max_len = _safe_int(INJECTION_GUARD.get("max_input_length", 0))
    if max_len > 0 and len(prompt) > max_len:
        return _rejected_prompt_error("max_length_exceeded")

    blocked_patterns = INJECTION_GUARD.get("blocked_patterns", [])
    if isinstance(blocked_patterns, list):
        for pattern in blocked_patterns:
            pat = str(pattern).strip().lower()
            if pat and pat in normalized_prompt:
                return _rejected_prompt_error("blocked_pattern")
    return None


def _degraded_result(model: str, error: Dict[str, Any]) -> GenerateTextResult:
    return GenerateTextResult(
        text="",
        model=model,
        provider=_PROVIDER,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        degraded=True,
        error=error,
    )


def _call_provider(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int | None,
) -> GenerateTextResult:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    response = client.chat.completions.create(**payload)
    text = _extract_text(response)
    input_tokens, output_tokens, total_tokens = _extract_usage(response)
    response_model = getattr(response, "model", None)
    model_used = (
        str(response_model).strip() if isinstance(response_model, str) else model
    )
    return GenerateTextResult(
        text=text,
        model=model_used,
        provider=_PROVIDER,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        degraded=False,
        error=None,
    )


def generate_text(
    prompt: str,
    agent_key: str,
    model: str = _PRIMARY_MODEL,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerateTextResult:
    """Generate text using OpenAI with guarded retries and safe fallback behavior."""
    agent = str(agent_key).strip()
    if agent not in RETRY_POLICY:
        return _degraded_result(
            model=model or _PRIMARY_MODEL,
            error={
                "code": "invalid_agent_key",
                "message": "Unknown agent key for retry policy.",
                "provider": _PROVIDER,
                "recoverable": False,
            },
        )

    raw_prompt = str(prompt or "")
    safe_prompt = _sanitize_prompt(raw_prompt)
    if not safe_prompt:
        return _degraded_result(
            model=model or _PRIMARY_MODEL,
            error=_rejected_prompt_error("empty_prompt"),
        )

    blocked_error = _validate_prompt(safe_prompt)
    if blocked_error is not None:
        return _degraded_result(model=model or _PRIMARY_MODEL, error=blocked_error)

    if not live_provider_calls_enabled():
        return _degraded_result(
            model=model or _PRIMARY_MODEL,
            error={
                "code": "live_calls_disabled",
                "message": "Live provider calls are disabled by CONTENTBLITZ_ENABLE_LIVE_CALLS.",
                "provider": _PROVIDER,
                "recoverable": False,
            },
        )

    api_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return _degraded_result(
            model=model or _PRIMARY_MODEL,
            error={
                "code": "configuration_error",
                "message": "OPENAI_API_KEY is not configured.",
                "provider": _PROVIDER,
                "recoverable": False,
            },
        )

    client = _build_openai_client(api_key=api_key)
    attempts_per_model = _safe_int(RETRY_POLICY.get(agent, 0), default=0) + 1

    primary_model = str(model).strip() or _PRIMARY_MODEL
    models_to_try = [primary_model]
    if primary_model != _FALLBACK_MODEL:
        models_to_try.append(_FALLBACK_MODEL)

    last_error: Optional[Dict[str, Any]] = None
    for model_name in models_to_try:
        for _ in range(attempts_per_model):
            try:
                return _call_provider(
                    client,
                    model=model_name,
                    prompt=safe_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - deterministic via mocks in tests
                last_error = _normalize_provider_error(exc)

    return _degraded_result(
        model=models_to_try[-1],
        error={
            "code": "provider_failure",
            "message": "Text generation failed after retries and fallback.",
            "provider": _PROVIDER,
            "recoverable": True,
            "attempts_per_model": attempts_per_model,
            "models_attempted": models_to_try,
            "last_error": last_error,
        },
    )


__all__ = ["GenerateTextResult", "generate_text"]
