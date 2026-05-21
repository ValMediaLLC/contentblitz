"""Provider-backed text generation tool with deterministic safety behavior."""

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
from contentblitz.core.model_policy import (
    ProviderModelSelection,
    resolve_text_provider_model,
)
from contentblitz.core.observability import safe_tool_metadata, start_tool_span

try:  # pragma: no cover - optional dependency availability is environment-specific.
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]

_OPENAI_PROVIDER = "openai"
_ANTHROPIC_PROVIDER = "anthropic"
_SUPPORTED_PROVIDERS = {_OPENAI_PROVIDER, _ANTHROPIC_PROVIDER}
_ANTHROPIC_STABLE_SYSTEM_PROMPT = (
    "You are ContentBlitz text generation assistant. Return plain text only."
)
_SAFE_PROVIDER_ERROR_CODES = {
    "quota_exceeded",
    "authentication_failed",
    "rate_limited",
    "provider_unavailable",
    "empty_provider_response",
    "unknown_provider_error",
    "configuration_error",
}
_RETRY_POLICY_AGENT_ALIASES = {
    "clarification": "query_handler",
}
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class GenerateTextResult:
    """Normalized text-generation result returned by the provider tool."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    degraded: bool
    error: Optional[Dict[str, Any]]


def _default_selection(agent_key: str) -> ProviderModelSelection:
    return resolve_text_provider_model(agent_key or "default", near_budget=False)


def _retry_policy_agent_key(agent_key: str) -> str:
    normalized = str(agent_key).strip()
    if not normalized:
        return normalized
    return _RETRY_POLICY_AGENT_ALIASES.get(normalized, normalized)


def _provider_api_key_env_name(provider: str) -> str:
    if provider == _ANTHROPIC_PROVIDER:
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def _read_bool_env(var_name: str, *, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    token = str(raw).strip().lower()
    if token in _TRUE_ENV_VALUES:
        return True
    if token in _FALSE_ENV_VALUES:
        return False
    return default


def _anthropic_prompt_cache_enabled() -> bool:
    return _read_bool_env("CONTENTBLITZ_ENABLE_ANTHROPIC_PROMPT_CACHE", default=False)


def _build_openai_client(api_key: str) -> OpenAI:
    """Factory function to enable deterministic client mocking in tests."""
    return OpenAI(api_key=api_key)


def _build_anthropic_client(api_key: str) -> Any:
    """Factory function to enable deterministic client mocking in tests."""
    if Anthropic is None:
        raise RuntimeError("anthropic_sdk_unavailable")
    return Anthropic(api_key=api_key)


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return default


def _safe_error_class(exc: Exception) -> str:
    class_name = exc.__class__.__name__.strip().lower()
    return class_name or "unknown_error"


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


def _extract_openai_text(response: Any) -> str:
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


def _extract_anthropic_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    fragments: list[str] = []
    for block in content:
        block_type = ""
        block_text = ""
        if isinstance(block, Mapping):
            block_type = str(block.get("type", "")).strip().lower()
            block_text = str(block.get("text", "")).strip()
        else:
            block_type = str(getattr(block, "type", "")).strip().lower()
            block_text = str(getattr(block, "text", "")).strip()
        if block_type == "text" and block_text:
            fragments.append(block_text)
    return "\n".join(fragments).strip()


def _extract_openai_usage(response: Any) -> tuple[int, int, int]:
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


def _extract_anthropic_usage(response: Any) -> tuple[int, int, int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0, 0, 0
    input_tokens = _safe_int(getattr(usage, "input_tokens", 0))
    output_tokens = _safe_int(getattr(usage, "output_tokens", 0))
    total_tokens = input_tokens + output_tokens
    cache_creation = _safe_int(getattr(usage, "cache_creation_input_tokens", 0))
    cache_read = _safe_int(getattr(usage, "cache_read_input_tokens", 0))
    return input_tokens, output_tokens, total_tokens, cache_creation, cache_read


def _normalize_openai_provider_error(exc: Exception) -> Dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    status_value = status_code if isinstance(status_code, int) else None
    message_text = str(exc).lower()

    if isinstance(exc, AuthenticationError):
        return {
            "code": "authentication_failed",
            "message": "Text generation provider authentication failed.",
            "provider": _OPENAI_PROVIDER,
            "status_code": status_value,
            "error_class": _safe_error_class(exc),
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
            "provider": _OPENAI_PROVIDER,
            "status_code": status_value,
            "error_class": _safe_error_class(exc),
            "recoverable": True,
        }
    if isinstance(exc, APIConnectionError):
        return {
            "code": "provider_unavailable",
            "message": "Text generation provider is temporarily unavailable.",
            "provider": _OPENAI_PROVIDER,
            "status_code": status_value,
            "error_class": _safe_error_class(exc),
            "recoverable": True,
        }
    if isinstance(exc, BadRequestError):
        return {
            "code": "unknown_provider_error",
            "message": "Text generation provider request failed safely.",
            "provider": _OPENAI_PROVIDER,
            "status_code": status_value,
            "error_class": _safe_error_class(exc),
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
            "provider": _OPENAI_PROVIDER,
            "status_code": status_value,
            "error_class": _safe_error_class(exc),
            "recoverable": True,
        }

    return {
        "code": "unknown_provider_error",
        "message": "Text generation provider request failed safely.",
        "provider": _OPENAI_PROVIDER,
        "status_code": status_value,
        "error_class": _safe_error_class(exc),
        "recoverable": True,
    }


def _normalize_anthropic_provider_error(exc: Exception) -> Dict[str, Any]:
    message_text = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    status_value = status_code if isinstance(status_code, int) else None
    class_name = exc.__class__.__name__.lower()

    if (
        "anthropic_sdk_unavailable" in message_text
        or "modulenotfounderror" in class_name
    ):
        code = "configuration_error"
        message = "Anthropic provider is not available in this environment."
        recoverable = False
    elif "authentication" in class_name or status_value in {401, 403}:
        code = "authentication_failed"
        message = "Text generation provider authentication failed."
        recoverable = False
    elif "ratelimit" in class_name or status_value == 429:
        is_quota_error = "quota" in message_text or "billing" in message_text
        code = "quota_exceeded" if is_quota_error else "rate_limited"
        message = (
            "Text generation provider is unavailable or quota-limited."
            if is_quota_error
            else "Text generation provider is rate-limited."
        )
        recoverable = True
    elif "timeout" in class_name or "connection" in class_name or (
        isinstance(status_value, int) and status_value >= 500
    ):
        code = "provider_unavailable"
        message = "Text generation provider is temporarily unavailable."
        recoverable = True
    elif status_value == 400:
        code = "configuration_error"
        message = "Text generation provider request failed safely."
        recoverable = False
    else:
        code = "unknown_provider_error"
        message = "Text generation provider request failed safely."
        recoverable = True

    return {
        "code": code,
        "message": message,
        "provider": _ANTHROPIC_PROVIDER,
        "status_code": status_value,
        "error_class": _safe_error_class(exc),
        "recoverable": recoverable,
    }


def _normalize_provider_error(exc: Exception, provider: str) -> Dict[str, Any]:
    if provider == _ANTHROPIC_PROVIDER:
        return _normalize_anthropic_provider_error(exc)
    return _normalize_openai_provider_error(exc)


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
    cleaned = prompt.replace("\x00", " ").strip()
    return cleaned


def _validate_prompt(prompt: str, provider: str) -> Optional[Dict[str, Any]]:
    normalized_prompt = prompt.lower()
    max_len = _safe_int(INJECTION_GUARD.get("max_input_length", 0))
    if max_len > 0 and len(prompt) > max_len:
        return _rejected_prompt_error("max_length_exceeded", provider)

    blocked_patterns = INJECTION_GUARD.get("blocked_patterns", [])
    if isinstance(blocked_patterns, list):
        for pattern in blocked_patterns:
            pat = str(pattern).strip().lower()
            if pat and pat in normalized_prompt:
                return _rejected_prompt_error("blocked_pattern", provider)
    return None


def _degraded_result(
    *,
    provider: str,
    model: str,
    error: Dict[str, Any],
) -> GenerateTextResult:
    return GenerateTextResult(
        text="",
        model=model,
        provider=provider,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        degraded=True,
        error=error,
    )


def _with_safe_diagnostics(
    *,
    error: Mapping[str, Any] | None,
    requested_provider: str,
    requested_model: str,
    fallback_provider: str,
    fallback_model: str,
    provider_models_attempted: list[str] | None = None,
    last_error: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = dict(error or {})
    diagnostics["requested_provider"] = str(requested_provider).strip().lower()
    diagnostics["requested_model"] = str(requested_model).strip()
    diagnostics["fallback_provider"] = str(fallback_provider).strip().lower()
    diagnostics["fallback_model"] = str(fallback_model).strip()
    if provider_models_attempted:
        diagnostics["provider_models_attempted"] = [
            str(item).strip()
            for item in provider_models_attempted
            if str(item).strip()
        ]

    safe_last_error: Dict[str, Any] = {}
    if isinstance(last_error, Mapping):
        code = str(last_error.get("code", "")).strip().lower()
        provider = str(last_error.get("provider", "")).strip().lower()
        status_code = _safe_int(last_error.get("status_code"), default=-1)
        error_class = str(last_error.get("error_class", "")).strip().lower()
        if code:
            safe_last_error["code"] = code
        if provider:
            safe_last_error["provider"] = provider
        if status_code >= 0:
            safe_last_error["status_code"] = status_code
        if error_class:
            safe_last_error["error_class"] = error_class
    if safe_last_error:
        diagnostics["last_error"] = safe_last_error

    return diagnostics


def _fallback_used(
    *,
    requested_provider: str,
    requested_model: str,
    result: GenerateTextResult,
) -> bool:
    requested_provider_text = str(requested_provider).strip().lower()
    requested_model_text = str(requested_model).strip()
    if not result.degraded:
        return (
            str(result.provider).strip().lower() != requested_provider_text
            or str(result.model).strip() != requested_model_text
        )
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
    requested_provider: str,
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
        "provider": requested_provider,
        "model": requested_model,
        "agent_key": agent_key,
        "retry_attempt": max(0, int(retry_attempt)),
        "retry_exhausted": bool(retry_exhausted),
        "duration_ms": duration_ms,
    }
    outputs: Dict[str, Any] = {}

    if result is not None:
        fallback_used = _fallback_used(
            requested_provider=requested_provider,
            requested_model=requested_model,
            result=result,
        )
        metadata.update(
            {
                "provider": result.provider,
                "model": result.model,
                "final_model": result.model,
                "degraded": result.degraded,
                "fallback_used": fallback_used,
                "input_token_count": result.input_tokens,
                "output_token_count": result.output_tokens,
                "total_token_count": result.total_tokens,
                "cache_creation_input_tokens": result.cache_creation_input_tokens,
                "cache_read_input_tokens": result.cache_read_input_tokens,
            }
        )
        if fallback_used:
            metadata["fallback_provider"] = result.provider
            metadata["fallback_model"] = result.model
            metadata["fallback_reason"] = "model_or_provider_fallback"
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


def _call_openai_provider(
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
    text = _extract_openai_text(response)
    input_tokens, output_tokens, total_tokens = _extract_openai_usage(response)
    response_model = getattr(response, "model", None)
    model_used = (
        str(response_model).strip() if isinstance(response_model, str) else model
    )
    if not text:
        return _degraded_result(
            provider=_OPENAI_PROVIDER,
            model=model_used,
            error={
                "code": "empty_provider_response",
                "message": "Text generation provider returned no usable output.",
                "provider": _OPENAI_PROVIDER,
                "recoverable": True,
            },
        )
    return GenerateTextResult(
        text=text,
        model=model_used,
        provider=_OPENAI_PROVIDER,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        degraded=False,
        error=None,
    )


def _call_anthropic_provider(
    client: Any,
    *,
    model: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int | None,
) -> GenerateTextResult:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens) if max_tokens is not None else 1024,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)

    if _anthropic_prompt_cache_enabled():
        payload["system"] = [
            {
                "type": "text",
                "text": _ANTHROPIC_STABLE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        payload["system"] = _ANTHROPIC_STABLE_SYSTEM_PROMPT

    response = client.messages.create(**payload)
    text = _extract_anthropic_text(response)
    (
        input_tokens,
        output_tokens,
        total_tokens,
        cache_creation_input_tokens,
        cache_read_input_tokens,
    ) = _extract_anthropic_usage(response)
    response_model = getattr(response, "model", None)
    model_used = (
        str(response_model).strip() if isinstance(response_model, str) else model
    )

    if not text:
        return _degraded_result(
            provider=_ANTHROPIC_PROVIDER,
            model=model_used,
            error={
                "code": "empty_provider_response",
                "message": "Text generation provider returned no usable output.",
                "provider": _ANTHROPIC_PROVIDER,
                "recoverable": True,
            },
        )

    return GenerateTextResult(
        text=text,
        model=model_used,
        provider=_ANTHROPIC_PROVIDER,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        degraded=False,
        error=None,
    )


def _build_provider_client(provider: str, api_key: str) -> Any:
    if provider == _OPENAI_PROVIDER:
        return _build_openai_client(api_key=api_key)
    if provider == _ANTHROPIC_PROVIDER:
        return _build_anthropic_client(api_key=api_key)
    raise RuntimeError("unsupported_text_provider")


def _call_provider(
    *,
    client: Any,
    provider: str,
    model: str,
    prompt: str,
    temperature: float | None,
    max_tokens: int | None,
) -> GenerateTextResult:
    if provider == _OPENAI_PROVIDER:
        return _call_openai_provider(
            client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == _ANTHROPIC_PROVIDER:
        return _call_anthropic_provider(
            client,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _degraded_result(
        provider=provider,
        model=model,
        error={
            "code": "configuration_error",
            "message": "Configured text provider is not supported.",
            "provider": provider,
            "recoverable": False,
        },
    )


def _build_provider_candidates(
    *,
    requested_provider: str,
    requested_model: str,
    fallback_provider: str,
    fallback_model: str,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for provider, model in (
        (requested_provider, requested_model),
        (fallback_provider, fallback_model),
    ):
        normalized_provider = str(provider).strip().lower()
        normalized_model = str(model).strip()
        if (
            normalized_provider not in _SUPPORTED_PROVIDERS
            or not normalized_model
            or (normalized_provider, normalized_model) in candidates
        ):
            continue
        candidates.append((normalized_provider, normalized_model))
    return candidates


def generate_text(
    prompt: str,
    agent_key: str,
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerateTextResult:
    """Generate text with guarded retries and safe fallback behavior."""
    agent = str(agent_key).strip()
    agent_selection = _default_selection(agent or "default")
    requested_provider = agent_selection.provider
    requested_model = str(model).strip() if model is not None else ""
    if not requested_model:
        requested_model = agent_selection.model
    fallback_provider = agent_selection.fallback_provider or requested_provider
    fallback_model = agent_selection.fallback_model or requested_model

    started_at = time.perf_counter()
    tool_span = start_tool_span(
        "generate_text",
        metadata={
            "provider": requested_provider,
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
            requested_provider=requested_provider,
            requested_model=requested_model,
            agent_key=agent,
            retry_attempt=retry_attempt,
            retry_exhausted=retry_exhausted,
            result=result,
        )
        return result

    try:
        retry_policy_agent = _retry_policy_agent_key(agent)
        if retry_policy_agent not in RETRY_POLICY:
            return _finalize(
                _degraded_result(
                    provider=requested_provider,
                    model=requested_model,
                    error=_with_safe_diagnostics(
                        error={
                            "code": "invalid_agent_key",
                            "message": "Unknown agent key for retry policy.",
                            "provider": requested_provider,
                            "recoverable": False,
                        },
                        requested_provider=requested_provider,
                        requested_model=requested_model,
                        fallback_provider=fallback_provider,
                        fallback_model=fallback_model,
                    ),
                )
            )

        raw_prompt = str(prompt or "")
        safe_prompt = _sanitize_prompt(raw_prompt)
        if not safe_prompt:
            return _finalize(
                _degraded_result(
                    provider=requested_provider,
                    model=requested_model,
                    error=_with_safe_diagnostics(
                        error=_rejected_prompt_error(
                            "empty_prompt",
                            requested_provider,
                        ),
                        requested_provider=requested_provider,
                        requested_model=requested_model,
                        fallback_provider=fallback_provider,
                        fallback_model=fallback_model,
                    ),
                )
            )

        blocked_error = _validate_prompt(safe_prompt, requested_provider)
        if blocked_error is not None:
            return _finalize(
                _degraded_result(
                    provider=requested_provider,
                    model=requested_model,
                    error=_with_safe_diagnostics(
                        error=blocked_error,
                        requested_provider=requested_provider,
                        requested_model=requested_model,
                        fallback_provider=fallback_provider,
                        fallback_model=fallback_model,
                    ),
                )
            )

        if not live_provider_calls_enabled():
            return _finalize(
                _degraded_result(
                    provider=requested_provider,
                    model=requested_model,
                    error=_with_safe_diagnostics(
                        error={
                            "code": "live_calls_disabled",
                            "message": (
                                "Live provider calls are disabled by "
                                "CONTENTBLITZ_ENABLE_LIVE_CALLS."
                            ),
                            "provider": requested_provider,
                            "recoverable": False,
                        },
                        requested_provider=requested_provider,
                        requested_model=requested_model,
                        fallback_provider=fallback_provider,
                        fallback_model=fallback_model,
                    ),
                )
            )

        attempts_per_model = (
            _safe_int(RETRY_POLICY.get(retry_policy_agent, 0), default=0) + 1
        )
        attempt_counter = 0
        candidates = _build_provider_candidates(
            requested_provider=requested_provider,
            requested_model=requested_model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
        )
        attempted_models: list[str] = []
        attempted_provider_models: list[str] = []
        provider_clients: dict[str, Any] = {}
        last_error: Optional[Dict[str, Any]] = None

        for provider_name, model_name in candidates:
            attempted_models.append(model_name)
            attempted_provider_models.append(f"{provider_name}:{model_name}")
            api_key_env_name = _provider_api_key_env_name(provider_name)
            api_key = str(os.getenv(api_key_env_name, "")).strip()
            if not api_key:
                last_error = {
                    "code": "configuration_error",
                    "message": "Provider credentials are not configured.",
                    "provider": provider_name,
                    "status_code": None,
                    "error_class": "missing_api_key",
                    "recoverable": provider_name != requested_provider,
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

            for _ in range(attempts_per_model):
                attempt_counter += 1
                try:
                    provider_result = _call_provider(
                        client=provider_clients[provider_name],
                        provider=provider_name,
                        model=model_name,
                        prompt=safe_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:  # pragma: no cover
                    last_error = _normalize_provider_error(exc, provider_name)
                    continue

                if provider_result.degraded:
                    error_payload = _with_safe_diagnostics(
                        error=provider_result.error,
                        requested_provider=requested_provider,
                        requested_model=requested_model,
                        fallback_provider=fallback_provider,
                        fallback_model=fallback_model,
                        provider_models_attempted=attempted_provider_models,
                    )
                    return _finalize(
                        GenerateTextResult(
                            text=provider_result.text,
                            model=provider_result.model,
                            provider=provider_result.provider,
                            input_tokens=provider_result.input_tokens,
                            output_tokens=provider_result.output_tokens,
                            total_tokens=provider_result.total_tokens,
                            cache_creation_input_tokens=(
                                provider_result.cache_creation_input_tokens
                            ),
                            cache_read_input_tokens=(
                                provider_result.cache_read_input_tokens
                            ),
                            degraded=True,
                            error=error_payload,
                        ),
                        retry_attempt=attempt_counter,
                        retry_exhausted=False,
                    )

                return _finalize(
                    provider_result,
                    retry_attempt=attempt_counter,
                    retry_exhausted=False,
                )

        final_provider = requested_provider
        final_model = requested_model
        if candidates:
            final_provider, final_model = candidates[-1]

        safe_error_code = (
            str(last_error.get("code", "")).strip().lower()
            if isinstance(last_error, Mapping)
            else ""
        )
        if safe_error_code not in _SAFE_PROVIDER_ERROR_CODES:
            safe_error_code = "unknown_provider_error"

        return _finalize(
            _degraded_result(
                provider=final_provider,
                model=final_model,
                error=_with_safe_diagnostics(
                    error={
                        "code": safe_error_code,
                        "message": (
                            "Text generation provider is unavailable or quota-limited."
                        ),
                        "provider": final_provider,
                        "recoverable": True,
                        "attempts_per_model": attempts_per_model,
                        "models_attempted": attempted_models,
                        "provider_models_attempted": attempted_provider_models,
                    },
                    requested_provider=requested_provider,
                    requested_model=requested_model,
                    fallback_provider=fallback_provider,
                    fallback_model=fallback_model,
                    provider_models_attempted=attempted_provider_models,
                    last_error=last_error,
                ),
            ),
            retry_attempt=attempt_counter,
            retry_exhausted=True,
        )
    except Exception as error:
        _finish_tool_span(
            span=tool_span,
            started_at=started_at,
            requested_provider=requested_provider,
            requested_model=requested_model,
            agent_key=agent,
            retry_attempt=0,
            retry_exhausted=False,
            error=error,
        )
        raise


__all__ = ["GenerateTextResult", "generate_text"]
