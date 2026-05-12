"""Deterministic prompt-injection detection and sanitization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Tuple

_COMMAND_VERBS_RE = re.compile(
    r"\b(?:reveal|show|display|print|dump|output|expose)\b",
    flags=re.IGNORECASE,
)
_SYSTEM_PROMPT_OBJECT_RE = re.compile(
    r"\b(?:system|hidden|internal|developer)[\W_]*(?:prompts?|messages?|instructions?)\b",
    flags=re.IGNORECASE,
)
_API_KEY_OBJECT_RE = re.compile(
    r"\bapi[\W_]*keys?\b",
    flags=re.IGNORECASE,
)
_SECRETS_OBJECT_RE = re.compile(
    r"\bsecrets?\b",
    flags=re.IGNORECASE,
)
_ENVIRONMENT_OBJECT_RE = re.compile(
    r"\b(?:environment|env)[\W_]*(?:variables?|vars?)\b",
    flags=re.IGNORECASE,
)
_SIGNAL_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"\bignore[\W_]*(?:all|any|previous)?[\W_]*instructions?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "reveal_system_prompt",
        re.compile(
            r"\b(?:reveal|show|display|print|dump|output|expose)[\W_]*(?:the[\W_]*)?"
            r"(?:system|hidden|internal|developer)[\W_]*(?:prompts?|messages?|instructions?)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "output_api_keys",
        re.compile(
            r"\b(?:reveal|show|display|print|dump|output|expose)[\W_]*(?:the[\W_]*)?"
            r"api[\W_]*keys?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "dump_secrets",
        re.compile(
            r"\b(?:dump|reveal|show|display|print|output|expose)[\W_]*(?:the[\W_]*)?"
            r"secrets?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "environment_variables",
        re.compile(
            r"\b(?:print|show|display|dump|output|reveal|expose)[\W_]*(?:the[\W_]*)?"
            r"(?:environment|env)[\W_]*(?:variables?|vars?)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "developer_message",
        re.compile(
            r"\bdeveloper[\W_]*messages?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "system_message",
        re.compile(
            r"\bsystem[\W_]*messages?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "bypass_guardrails",
        re.compile(
            r"\bbypass[\W_]*(?:guardrails?|protections?|safeguards?)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "disable_safety",
        re.compile(
            r"\bdisable[\W_]*(?:safety|guardrails?|protections?|safeguards?)\b",
            flags=re.IGNORECASE,
        ),
    ),
)
_RESIDUAL_UNSAFE_FRAGMENT_PATTERNS: Tuple[re.Pattern[str], ...] = (
    _SYSTEM_PROMPT_OBJECT_RE,
    _API_KEY_OBJECT_RE,
    _SECRETS_OBJECT_RE,
    _ENVIRONMENT_OBJECT_RE,
)

_MULTISPACE_RE = re.compile(r"\s+")
_DANGLING_CONNECTOR_RE = re.compile(
    r"(?:\b(?:and|or|then|also)\b[\s,;:._-]*)+$",
    flags=re.IGNORECASE,
)
_LEADING_CONNECTOR_RE = re.compile(
    r"^(?:[\s,;:._-]*\b(?:and|or|then|also)\b)+[\s,;:._-]*",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptInjectionResult:
    detected: bool
    signals: List[str]
    sanitized_query: str


def _safe_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _cleanup_query(query: str) -> str:
    cleaned = query.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    cleaned = _MULTISPACE_RE.sub(" ", cleaned).strip(" \t,;:.-!?")
    cleaned = _DANGLING_CONNECTOR_RE.sub("", cleaned).strip(" \t,;:.-!?")
    cleaned = _LEADING_CONNECTOR_RE.sub("", cleaned).strip(" \t,;:.-!?")
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        token = _safe_text(value).lower()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def analyze_prompt_injection(query: str) -> PromptInjectionResult:
    """
    Detect obvious prompt-injection patterns and return a sanitized query.

    This is intentionally lightweight and deterministic.
    """
    raw = _safe_text(query)
    if not raw:
        return PromptInjectionResult(detected=False, signals=[], sanitized_query="")

    sanitized = raw
    matched_signals: List[str] = []
    for signal, pattern in _SIGNAL_RULES:
        if pattern.search(raw):
            matched_signals.append(signal)
            sanitized = pattern.sub(" ", sanitized)

    high_risk_context = bool(
        _COMMAND_VERBS_RE.search(raw)
        or any(
            signal in matched_signals
            for signal in (
                "ignore_instructions",
                "reveal_system_prompt",
                "bypass_guardrails",
                "disable_safety",
                "developer_message",
                "system_message",
            )
        )
    )
    if high_risk_context and _API_KEY_OBJECT_RE.search(raw):
        matched_signals.append("output_api_keys")
    if high_risk_context and _SECRETS_OBJECT_RE.search(raw):
        matched_signals.append("dump_secrets")
    if high_risk_context and _ENVIRONMENT_OBJECT_RE.search(raw):
        matched_signals.append("environment_variables")

    if matched_signals:
        for pattern in _RESIDUAL_UNSAFE_FRAGMENT_PATTERNS:
            sanitized = pattern.sub(" ", sanitized)

    cleaned = _cleanup_query(sanitized)
    signals = _ordered_unique(matched_signals)
    detected = len(signals) > 0
    if not detected:
        cleaned = raw
    return PromptInjectionResult(
        detected=detected,
        signals=signals,
        sanitized_query=cleaned,
    )
