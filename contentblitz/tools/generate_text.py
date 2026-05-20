"""OpenAI-backed text generation tool with deterministic safety behavior."""

from __future__ import annotations

import os
import time
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

from contentblitz.config import (
    INJECTION_GUARD,
    RETRY_POLICY,
    live_provider_calls_enabled,
)
from contentblitz.core.observability import safe_tool_metadata, start_tool_span

_PROVIDER = "openai"
_PRIMARY_MODEL = "gpt-4o"
_FALLBACK_MODEL = "gpt-4o-mini"
_SAFE_PROVIDER_ERROR_CODES = {
    "quota_exceeded",
    "authentication_failed",
    "rate_limited",
    "provider_unavailable",
    "empty_provider_response",
    "unknown_provider_error",
}


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
    message_text = str(exc).lower()

    if isinstance(exc, AuthenticationError):
        return {
            "code": "authentication_failed",
            "message": "Text generation provider authentication failed.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": False,
        }
    if isinstance(exc, RateLimitError):
        is_quota_error = (
            "quota" in message_text
            or "insufficient_quota" in message_text
            or "billing" in message_text
        )
        return {
            "code": "quota_exceeded" if is_quota_error else "rate_limited",
            "message": (
                "Text generation provider is unavailable or quota-limited."
                if is_quota_error
                else "Text generation provider is rate-limited."
            ),
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    if isinstance(exc, APIConnectionError):
        return {
            "code": "provider_unavailable",
            "message": "Text generation provider is temporarily unavailable.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    if isinstance(exc, BadRequestError):
        return {
            "code": "unknown_provider_error",
            "message": "Text generation provider request failed safely.",
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }
    if isinstance(exc, APIError):
        is_unavailable = isinstance(status_value, int) and status_value >= 500
        return {
            "code": (
                "provider_unavailable"
                if is_unavailable
                else "unknown_provider_error"
            ),
            "message": (
                "Text generation provider is temporarily unavailable."
                if is_unavailable
                else "Text generation provider request failed safely."
            ),
            "provider": _PROVIDER,
            "status_code": status_value,
            "recoverable": True,
        }

    return {
        "code": "unknown_provider_error",
        "message": "Text generation provider request failed safely.",
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


def _fallback_used(
    *,
    requested_model: str,
    result: GenerateTextResult,
) -> bool:
    requested = str(requested_model).strip() or _PRIMARY_MODEL
    if not result.degraded:
        return str(result.model).strip() != requested
    if not isinstance(result.error, Mapping):
        return False
    attempted = result.error.get("models_attempted", [])
    if not isinstance(attempted, list):
        return False
    normalized = [str(item).strip() for item in attempted if str(item).strip()]
    return len(normalized) > 1


def _finish_tool_span(
    *,
    span: Any,
    started_at: float,
    requested_model: str,
    agent_key: str,
    retry_attempt: int = 0,
    retry_exhausted: bool = False,
    result: GenerateTextResult | None = None,
    error: BaseException | None = None,
) -> None:
    duration_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    metadata: Dict[str, Any] = {
        "tool_name": "generate_text",
        "provider": _PROVIDER,
        "model": requested_model,
        "agent_key": agent_key,
        "retry_attempt": max(0, int(retry_attempt)),
        "retry_exhausted": bool(retry_exhausted),
        "duration_ms": duration_ms,
    }
    outputs: Dict[str, Any] = {}

    if result is not None:
        fallback_used = _fallback_used(
            requested_model=requested_model,
            result=result,
        )
        metadata.update(
            {
                "model": result.model,
                "final_model": result.model,
                "degraded": result.degraded,
                "fallback_used": fallback_used,
                "input_token_count": result.input_tokens,
                "output_token_count": result.output_tokens,
                "total_token_count": result.total_tokens,
            }
        )
        if fallback_used:
            metadata["fallback_provider"] = _PROVIDER
            metadata["fallback_model"] = result.model
            metadata["fallback_reason"] = "model_fallback"
        if result.degraded and isinstance(result.error, Mapping):
            fallback_reason = str(result.error.get("code", "")).strip()
            if fallback_reason:
                metadata["fallback_reason"] = fallback_reason
        outputs = {
            "degraded": result.degraded,
            "provider": result.provider,
            "model": result.model,
            "retry_exhausted": bool(retry_exhausted),
        }
    elif error is not None:
        metadata.update(
            {
                "degraded": True,
                "fallback_used": False,
                "fallback_reason": "tool_exception",
            }
        )

    span.finish(
        metadata=safe_tool_metadata(metadata),
        outputs=outputs,
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
    if not text:
        return _degraded_result(
            model=model_used,
            error={
                "code": "empty_provider_response",
                "message": (
                    "Text generation provider returned no usable output."
                ),
                "provider": _PROVIDER,
                "recoverable": True,
            },
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
    requested_model = str(model).strip() or _PRIMARY_MODEL
    agent = str(agent_key).strip()
    started_at = time.perf_counter()
    tool_span = start_tool_span(
        "generate_text",
        metadata={
            "provider": _PROVIDER,
            "model": requested_model,
            "agent_key": agent,
        },
        inputs={"tool_name": "generate_text", "agent_key": agent},
    )

    def _finalize(
        result: GenerateTextResult,
        *,
        retry_attempt: int = 0,
        retry_exhausted: bool = False,
    ) -> GenerateTextResult:
        _finish_tool_span(
            span=tool_span,
            started_at=started_at,
            requested_model=requested_model,
            agent_key=agent,
            retry_attempt=retry_attempt,
            retry_exhausted=retry_exhausted,
            result=result,
        )
        return result

    try:
        if agent not in RETRY_POLICY:
            return _finalize(
                _degraded_result(
                    model=requested_model,
                    error={
                        "code": "invalid_agent_key",
                        "message": "Unknown agent key for retry policy.",
                        "provider": _PROVIDER,
                        "recoverable": False,
                    },
                )
            )

        raw_prompt = str(prompt or "")
        safe_prompt = _sanitize_prompt(raw_prompt)
        if not safe_prompt:
            return _finalize(
                _degraded_result(
                    model=requested_model,
                    error=_rejected_prompt_error("empty_prompt"),
                )
            )

        blocked_error = _validate_prompt(safe_prompt)
        if blocked_error is not None:
            return _finalize(
                _degraded_result(model=requested_model, error=blocked_error)
            )

        if not live_provider_calls_enabled():
            return _finalize(
                _degraded_result(
                    model=requested_model,
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
                    model=requested_model,
                    error={
                        "code": "configuration_error",
                        "message": "OPENAI_API_KEY is not configured.",
                        "provider": _PROVIDER,
                        "recoverable": False,
                    },
                )
            )

        client = _build_openai_client(api_key=api_key)
        attempts_per_model = _safe_int(RETRY_POLICY.get(agent, 0), default=0) + 1
        attempt_counter = 0

        models_to_try = [requested_model]
        if requested_model != _FALLBACK_MODEL:
            models_to_try.append(_FALLBACK_MODEL)

        last_error: Optional[Dict[str, Any]] = None
        for model_name in models_to_try:
            for _ in range(attempts_per_model):
                attempt_counter += 1
                try:
                    return _finalize(
                        _call_provider(
                            client,
                            model=model_name,
                            prompt=safe_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        retry_attempt=attempt_counter,
                        retry_exhausted=False,
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - deterministic via mocks in tests
                    last_error = _normalize_provider_error(exc)

        return _finalize(
            _degraded_result(
                model=models_to_try[-1],
                error={
                    "code": (
                        str(last_error.get("code", "")).strip().lower()
                        if isinstance(last_error, Mapping)
                        and str(last_error.get("code", "")).strip().lower()
                        in _SAFE_PROVIDER_ERROR_CODES
                        else "unknown_provider_error"
                    ),
                    "message": (
                        "Text generation provider is unavailable or quota-limited."
                    ),
                    "provider": _PROVIDER,
                    "recoverable": True,
                    "attempts_per_model": attempts_per_model,
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
            requested_model=requested_model,
            agent_key=agent,
            retry_attempt=0,
            retry_exhausted=False,
            error=error,
        )
        raise


__all__ = ["GenerateTextResult", "generate_text"]
