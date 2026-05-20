"""Observability configuration and safe tracing helpers."""

from __future__ import annotations

import hashlib
import os
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Protocol
from urllib.parse import urlparse

from contentblitz.config import (
    RETRY_POLICY,
    langsmith_api_key_present,
    langsmith_endpoint,
    langsmith_project,
    langsmith_tracing_requested,
)
from contentblitz.core.redaction import (
    MAX_TRACE_PREVIEW_CHARS,
    normalize_trace_error,
    sanitize_trace_value,
    summarize_text_content,
)

_STATUS_VALUES = {"pending", "running", "completed", "degraded", "failed", "skipped"}
_WORKFLOW_FAILURE_STATUSES = {
    "failed",
    "degraded",
    "partial_success",
    "completed_with_warnings",
}
_AUTHORITATIVE_NODE_SET = {
    "query_handler_node",
    "clarification_node",
    "research_agent_node",
    "content_strategist_node",
    "blog_writer_node",
    "linkedin_writer_node",
    "image_agent_node",
    "quality_validator_node",
    "retry_router_node",
    "output_assembler_node",
    "export_node",
    "error_handler_node",
}
_COST_COUNTER_KEYS = (
    "tokens_used_this_session",
    "search_queries_used_this_session",
    "image_generations_used_this_session",
    "total_retries_used_this_session",
)
_COST_CAP_KEYS = (
    "token_budget_per_session",
    "search_query_cap_per_session",
    "image_generation_cap_per_session",
    "max_total_retries_per_session",
)
_TOOL_TRACE_STRING_FIELDS = (
    "tool_name",
    "provider",
    "model",
    "agent_key",
    "final_model",
    "fallback_provider",
    "fallback_model",
    "fallback_reason",
)
_TOOL_TRACE_BOOL_FIELDS = (
    "degraded",
    "fallback_used",
    "image_url_present",
    "cache_hit",
    "cache_miss",
    "retry_exhausted",
    "budget_exceeded",
)
_TOOL_TRACE_INT_FIELDS = (
    "retry_attempt",
    "input_token_count",
    "output_token_count",
    "total_token_count",
    "result_count",
    "citation_available_count",
    "source_count",
    "image_output_count",
    "duration_ms",
)
_TRACE_SAMPLE_RATE_ENV = "CONTENTBLITZ_TRACE_SAMPLE_RATE"
_TRACE_FAILURE_SAMPLE_RATE_ENV = "CONTENTBLITZ_TRACE_FAILURE_SAMPLE_RATE"
_DEFAULT_TRACE_SAMPLE_RATE = 1.0
_DEFAULT_FAILURE_TRACE_SAMPLE_RATE = 1.0
_MAX_SOURCE_DOMAINS = 8
_MAX_EXPORT_FORMATS = 8
_MAX_RETRY_TARGETS = 6
_MAX_TOOL_INPUT_KEYS = 8
_TRACE_INPUT_INTENT_OPTIONS = ("blog", "linkedin", "image", "pdf", "md", "html", "docx")
_TRACE_INPUT_OUTPUT_INTENTS = {"blog", "linkedin", "image"}
_TRACE_INPUT_EXPORT_INTENT_MAP = {
    "markdown": "md",
    "md": "md",
    "html": "html",
    "pdf": "pdf",
    "docx": "docx",
    "word": "docx",
}
_ENV_STYLE_METADATA_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
_FORBIDDEN_ENV_METADATA_KEYS = {
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "LANGSMITH_API_KEY",
    "OPENAI_API_KEY",
    "SERP_API_KEY",
    "PERPLEXITY_API_KEY",
}
_SAMPLING_CONTEXT: ContextVar["_TraceSamplingContext | None"] = ContextVar(
    "contentblitz_trace_sampling_context",
    default=None,
)


@dataclass(frozen=True)
class ObservabilityConfig:
    """Public, secret-safe observability configuration."""

    tracing_requested: bool
    tracing_enabled: bool
    trace_sample_rate: float
    trace_failure_sample_rate: float
    endpoint: str
    project: str
    status: str
    message: str


@dataclass(frozen=True)
class TraceSamplingDecision:
    """Sampling outcome for success vs failure trace emission."""

    success_sampled: bool
    failure_sampled: bool
    session_seed: str


@dataclass(frozen=True)
class _TraceSamplingContext:
    allow_live_child_spans: bool


@dataclass
class _SamplingAwareTraceSpanHandle:
    """Workflow span handle that supports deterministic success/failure sampling."""

    delegate_tracer: WorkflowTracer
    start_metadata: Mapping[str, Any]
    decision: TraceSamplingDecision
    context_token: Token[_TraceSamplingContext | None] | None = None
    delegate_handle: TraceSpanHandle | None = None
    closed: bool = False

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self.closed:
            return
        try:
            is_failure = _is_failure_trace(metadata=metadata, error=error)
            should_sample = (
                self.decision.failure_sampled
                if is_failure
                else self.decision.success_sampled
            )
            if not should_sample:
                return
            if self.delegate_handle is None:
                self.delegate_handle = self.delegate_tracer.start_workflow(
                    metadata=self.start_metadata,
                )
            self.delegate_handle.finish(
                metadata=metadata,
                outputs=outputs,
                error=error,
            )
        finally:
            if self.context_token is not None:
                _SAMPLING_CONTEXT.reset(self.context_token)
            self.closed = True


class TraceSpanHandle(Protocol):
    """Handle for ending a trace span safely."""

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """End the span; never raise to callers."""


class WorkflowTracer(Protocol):
    """Tracer abstraction for workflow and node spans."""

    def start_workflow(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        """Start a workflow-level trace span."""

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        """Start a node-level trace span."""

    def start_tool(
        self,
        *,
        tool_name: str,
        metadata: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
    ) -> TraceSpanHandle:
        """Start a tool-level trace span."""


def _safe_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or text.lower() in {"none", "null"}:
        return ""
    return text


def _is_forbidden_env_metadata_key(key: str) -> bool:
    raw_key = _safe_text(key)
    if not raw_key:
        return False
    normalized_upper = raw_key.upper()
    if normalized_upper in _FORBIDDEN_ENV_METADATA_KEYS:
        return True
    if normalized_upper.endswith("_API_KEY"):
        return True
    if raw_key == normalized_upper and _ENV_STYLE_METADATA_KEY_RE.fullmatch(raw_key):
        return True
    return False


def _strip_unsafe_env_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, nested_value in value.items():
            key_text = str(key)
            if _is_forbidden_env_metadata_key(key_text):
                continue
            cleaned[key_text] = _strip_unsafe_env_metadata(nested_value)
        return cleaned
    if isinstance(value, list):
        return [_strip_unsafe_env_metadata(item) for item in value]
    return value


def _safe_endpoint_host(endpoint: str) -> str:
    candidate = _safe_text(endpoint)
    if not candidate:
        return "unknown"
    parsed = urlparse(candidate)
    host = _safe_text(parsed.hostname)
    if host:
        return host
    if "://" not in candidate:
        reparsed = urlparse(f"https://{candidate}")
        host = _safe_text(reparsed.hostname)
        if host:
            return host
    return "unknown"


def _safe_observability_summary_metadata() -> dict[str, Any]:
    config = build_observability_config()
    return {
        "tracing_enabled": config.tracing_enabled,
        "provider": "langsmith",
        "project_name": _safe_text(config.project),
        "endpoint_host": _safe_endpoint_host(config.endpoint),
    }


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return None


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _safe_text(str(item))
        if not text:
            continue
        token = text.lower()
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
    return cleaned


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _safe_sample_rate(value: Any, *, default: float) -> float:
    candidate = _safe_float(value)
    if candidate is None:
        return default
    if candidate < 0.0 or candidate > 1.0:
        return default
    return candidate


def _read_sample_rate_env(var_name: str, *, default: float) -> float:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return _safe_sample_rate(raw, default=default)


def _sampling_seed(metadata: Mapping[str, Any]) -> str:
    session_id = _safe_text(metadata.get("session_id"))
    if session_id:
        return session_id
    requested_outputs = metadata.get("requested_outputs", [])
    routing_decision = _safe_text(metadata.get("routing_decision"))
    return f"{requested_outputs}|{routing_decision}|contentblitz"


def _deterministic_sample_value(seed: str, suffix: str = "") -> float:
    token = f"{seed}:{suffix}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(token).hexdigest()[:16]
    max_uint64 = float(16**16 - 1)
    return int(digest, 16) / max_uint64 if max_uint64 > 0 else 0.0


def _build_sampling_decision(
    *,
    metadata: Mapping[str, Any],
    config: ObservabilityConfig,
) -> TraceSamplingDecision:
    seed = _sampling_seed(metadata)
    success_value = _deterministic_sample_value(seed, "success")
    failure_value = _deterministic_sample_value(seed, "failure")
    return TraceSamplingDecision(
        success_sampled=success_value < config.trace_sample_rate,
        failure_sampled=failure_value < config.trace_failure_sample_rate,
        session_seed=seed,
    )


def _safe_content_preview(value: Any) -> str:
    preview = summarize_text_content(value, max_preview_chars=MAX_TRACE_PREVIEW_CHARS)
    return _safe_text(preview.get("preview"))


def _safe_workflow_trace_intent(metadata: Mapping[str, Any]) -> list[str]:
    requested_outputs = _safe_string_list(metadata.get("requested_outputs", []))
    export_formats = _safe_string_list(metadata.get("export_formats_requested", []))
    if not export_formats:
        export_metadata = metadata.get("export_metadata", {})
        if isinstance(export_metadata, Mapping):
            export_formats = _safe_string_list(
                export_metadata.get("formats_requested", [])
            )
    seen: set[str] = set()

    for output in requested_outputs:
        if output not in _TRACE_INPUT_OUTPUT_INTENTS or output in seen:
            continue
        seen.add(output)

    for export_format in export_formats:
        normalized = _TRACE_INPUT_EXPORT_INTENT_MAP.get(export_format, "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

    ordered_intent = [token for token in _TRACE_INPUT_INTENT_OPTIONS if token in seen]
    return ordered_intent


def safe_workflow_trace_inputs(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build safe workflow span inputs for LangSmith root traces."""
    intent = _safe_workflow_trace_intent(metadata)
    if not intent:
        return {}
    return {"intent": intent}


def _safe_draft_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    drafts = state.get("content_drafts", {})
    if not isinstance(drafts, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for channel in ("blog", "linkedin"):
        entry = drafts.get(channel)
        if not isinstance(entry, Mapping):
            continue
        body = _safe_text(entry.get("body"))
        text_summary = summarize_text_content(body)
        channel_summary: dict[str, Any] = {
            "length": text_summary["length"],
            "word_count": text_summary["word_count"],
            "line_count": text_summary["line_count"],
            "preview": text_summary["preview"],
            "sha256_prefix": text_summary["sha256_prefix"],
        }
        version = _safe_non_negative_int(entry.get("version"))
        if version is not None:
            channel_summary["version"] = version
        if channel == "linkedin":
            chars = _safe_non_negative_int(entry.get("character_count"))
            if chars is not None:
                channel_summary["character_count"] = chars
        summary[channel] = channel_summary
    return summary


def _safe_final_response_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    final_response = _safe_text(state.get("final_response"))
    if not final_response:
        return {}
    text_summary = summarize_text_content(final_response)
    section_count = len(
        [line for line in final_response.splitlines() if line.startswith("#")]
    )
    return {
        "length": text_summary["length"],
        "word_count": text_summary["word_count"],
        "line_count": text_summary["line_count"],
        "section_count": section_count,
        "preview": text_summary["preview"],
        "sha256_prefix": text_summary["sha256_prefix"],
    }


def _safe_research_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    research = state.get("research_data", {})
    if not isinstance(research, Mapping):
        return {}
    summary: dict[str, Any] = {
        "degraded": _safe_bool(research.get("degraded", False)),
    }
    for key in ("status", "quality"):
        value = _safe_text(research.get(key))
        if value:
            summary[key] = value
    for key in ("query_count", "source_count"):
        count = _safe_non_negative_int(research.get(key))
        if count is not None:
            summary[key] = count
    for key in ("keywords", "key_facts", "entities"):
        items = research.get(key, [])
        if isinstance(items, list):
            summary[f"{key}_count"] = len(items)
    summary_text = (
        _safe_text(research.get("synthesized_summary"))
        or _safe_text(research.get("summary"))
    )
    if summary_text:
        summary["summary_preview"] = _safe_content_preview(summary_text)
    return summary


def _extract_domain(url: str) -> str:
    lowered = _safe_text(url).lower()
    if not lowered:
        return ""
    if lowered.startswith("http://"):
        lowered = lowered[len("http://") :]
    elif lowered.startswith("https://"):
        lowered = lowered[len("https://") :]
    domain = lowered.split("/", 1)[0].strip()
    return domain


def _safe_sources_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    sources = state.get("sources", [])
    if not isinstance(sources, list):
        return {}
    total = 0
    citation_count = 0
    domains: list[str] = []
    seen: set[str] = set()
    for item in sources:
        if not isinstance(item, Mapping):
            continue
        total += 1
        if _safe_bool(item.get("citation_available", False)):
            citation_count += 1
        domain = _extract_domain(_safe_text(item.get("url")))
        if domain and domain not in seen:
            seen.add(domain)
            domains.append(domain)
    summary: dict[str, Any] = {
        "source_count": total,
        "citation_available_count": citation_count,
    }
    if domains:
        summary["domains"] = domains[:_MAX_SOURCE_DOMAINS]
    return summary


def _safe_image_output_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    image_outputs = state.get("image_outputs", [])
    if not isinstance(image_outputs, list):
        return {}
    providers: list[str] = []
    seen: set[str] = set()
    url_present = 0
    degraded_count = 0
    for item in image_outputs:
        if not isinstance(item, Mapping):
            continue
        status = _safe_text(item.get("status")).lower()
        if status in {"failed", "degraded"}:
            degraded_count += 1
        if _safe_text(item.get("url")) or _safe_text(item.get("local_path")):
            url_present += 1
        provider = _safe_text(item.get("provider"))
        if provider and provider not in seen:
            seen.add(provider)
            providers.append(provider)
    summary: dict[str, Any] = {
        "image_output_count": len(
            [item for item in image_outputs if isinstance(item, Mapping)]
        ),
        "image_url_present_count": url_present,
        "degraded_count": degraded_count,
    }
    if providers:
        summary["providers"] = providers[:_MAX_SOURCE_DOMAINS]
    return summary


def _safe_retry_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    retry_targets_raw = state.get("retry_targets", [])
    retry_targets = (
        _safe_string_list(retry_targets_raw)
        if isinstance(retry_targets_raw, list)
        else []
    )
    retry_counts = _safe_retry_counts_summary(state)
    max_used = max(retry_counts.values()) if retry_counts else 0
    cost_controls = state.get("cost_controls", {})
    max_total = None
    total_used = None
    budget_exceeded = False
    if isinstance(cost_controls, Mapping):
        max_total = _safe_non_negative_int(
            cost_controls.get("max_total_retries_per_session")
        )
        total_used = _safe_non_negative_int(
            cost_controls.get("total_retries_used_this_session")
        )
        budget_exceeded = _safe_bool(cost_controls.get("budget_exceeded", False))

    policy_exhausted = True
    for agent_key in RETRY_POLICY:
        used = retry_counts.get(agent_key, 0)
        if used < RETRY_POLICY[agent_key]:
            policy_exhausted = False
            break
    session_cap_exhausted = False
    if max_total is not None and total_used is not None:
        session_cap_exhausted = total_used >= max_total

    return {
        "retry_requested": _safe_bool(state.get("retry_requested", False)),
        "retry_target": _safe_text(state.get("retry_target")),
        "retry_target_count": len(retry_targets[:_MAX_RETRY_TARGETS]),
        "retry_targets": retry_targets[:_MAX_RETRY_TARGETS],
        "retry_attempt": max_used,
        "retry_exhausted": session_cap_exhausted or policy_exhausted,
        "budget_exceeded": budget_exceeded,
    }


def _is_failure_trace(
    *,
    metadata: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> bool:
    if error is not None:
        return True
    if not isinstance(metadata, Mapping):
        return False
    if _safe_bool(metadata.get("degraded_workflow_status", False)):
        return True
    status = _safe_text(metadata.get("workflow_status")).lower()
    if status in _WORKFLOW_FAILURE_STATUSES:
        return True
    return False


def _safe_retry_counts_summary(state: Mapping[str, Any]) -> dict[str, int]:
    retry_counts = state.get("retry_counts", {})
    if not isinstance(retry_counts, Mapping):
        return {}
    summary: dict[str, int] = {}
    for key, value in retry_counts.items():
        safe_key = _safe_text(str(key))
        safe_value = _safe_non_negative_int(value)
        if not safe_key or safe_value is None:
            continue
        summary[safe_key] = safe_value
    return summary


def _safe_cost_controls_summary(state: Mapping[str, Any]) -> dict[str, int | bool]:
    cost_controls = state.get("cost_controls", {})
    if not isinstance(cost_controls, Mapping):
        return {}
    summary: dict[str, int | bool] = {}
    for key in [*_COST_COUNTER_KEYS, *_COST_CAP_KEYS]:
        safe_value = _safe_non_negative_int(cost_controls.get(key))
        if safe_value is None:
            continue
        summary[key] = safe_value
    summary["budget_exceeded"] = _safe_bool(cost_controls.get("budget_exceeded", False))
    return summary


def _safe_export_formats(state: Mapping[str, Any]) -> list[str]:
    export_metadata = state.get("export_metadata", {})
    if not isinstance(export_metadata, Mapping):
        return []
    formats = _safe_string_list(export_metadata.get("formats_requested", []))
    return formats[:_MAX_EXPORT_FORMATS]


def _safe_format_list(value: Any, *, max_items: int = _MAX_EXPORT_FORMATS) -> list[str]:
    return _safe_string_list(value)[:max_items]


def _failed_export_formats(state: Mapping[str, Any]) -> list[str]:
    export_metadata = state.get("export_metadata", {})
    if not isinstance(export_metadata, Mapping):
        return []
    explicit_failed = _safe_format_list(
        export_metadata.get("failed_export_formats", [])
    )
    if explicit_failed:
        return explicit_failed
    export_status = export_metadata.get("export_status", {})
    if not isinstance(export_status, Mapping):
        return []
    failed: list[str] = []
    for fmt, status in export_status.items():
        safe_fmt = _safe_text(fmt).lower()
        safe_status = _safe_text(status).lower()
        if not safe_fmt or safe_status != "failed":
            continue
        if safe_fmt not in failed:
            failed.append(safe_fmt)
    return failed[:_MAX_EXPORT_FORMATS]


def _is_warning_export_log_entry(entry: Mapping[str, Any]) -> bool:
    code = _safe_text(entry.get("code")).lower()
    if code.endswith("_warning") or code == "warning":
        return True
    message = _safe_text(entry.get("message")).lower()
    return "warning" in message and "failed" not in message


def _export_error_count_from_log(export_metadata: Mapping[str, Any]) -> int:
    error_log = export_metadata.get("error_log", [])
    if not isinstance(error_log, list):
        return 0
    return sum(
        1
        for entry in error_log
        if isinstance(entry, Mapping) and not _is_warning_export_log_entry(entry)
    )


def _export_warning_count_from_log(export_metadata: Mapping[str, Any]) -> int:
    error_log = export_metadata.get("error_log", [])
    if not isinstance(error_log, list):
        return 0
    return sum(
        1
        for entry in error_log
        if isinstance(entry, Mapping) and _is_warning_export_log_entry(entry)
    )


def _safe_export_outcome_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    export_metadata = state.get("export_metadata", {})
    if not isinstance(export_metadata, Mapping):
        return {
            "requested_export_formats": [],
            "completed_export_formats": [],
            "failed_export_formats": [],
            "export_warning_count": 0,
            "export_error_count": 0,
        }

    requested_export_formats = _safe_format_list(
        export_metadata.get("requested_export_formats")
        or export_metadata.get("formats_requested", [])
    )
    completed_export_formats = _safe_format_list(
        export_metadata.get("completed_export_formats", [])
    )
    if not completed_export_formats:
        export_status = export_metadata.get("export_status", {})
        if isinstance(export_status, Mapping):
            completed_export_formats = [
                safe_fmt
                for fmt, status in export_status.items()
                for safe_fmt in [_safe_text(fmt).lower()]
                if safe_fmt and _safe_text(status).lower() == "completed"
            ][: _MAX_EXPORT_FORMATS]
    failed_export_formats = _failed_export_formats(state)

    raw_error_count = _safe_non_negative_int(export_metadata.get("export_error_count"))
    export_error_count = (
        raw_error_count
        if raw_error_count is not None
        else (
            len(failed_export_formats)
            if failed_export_formats
            else _export_error_count_from_log(export_metadata)
        )
    )
    raw_warning_count = _safe_non_negative_int(
        export_metadata.get("export_warning_count")
    )
    export_warning_count = (
        raw_warning_count
        if raw_warning_count is not None
        else _export_warning_count_from_log(export_metadata)
    )

    return {
        "requested_export_formats": requested_export_formats,
        "completed_export_formats": completed_export_formats,
        "failed_export_formats": failed_export_formats,
        "export_warning_count": export_warning_count,
        "export_error_count": export_error_count,
    }


def _safe_source_count(state: Mapping[str, Any]) -> int:
    sources = state.get("sources", [])
    if not isinstance(sources, list):
        return 0
    return len([item for item in sources if isinstance(item, Mapping)])


def _safe_image_output_count(state: Mapping[str, Any]) -> int:
    image_outputs = state.get("image_outputs", [])
    if not isinstance(image_outputs, list):
        return 0
    return len([item for item in image_outputs if isinstance(item, Mapping)])


def _safe_error_summary(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors = state.get("errors", [])
    if not isinstance(errors, list):
        return []
    summaries: list[dict[str, Any]] = []
    for item in errors[:3]:
        summaries.append(normalize_trace_error(item))
    return summaries


def _is_fallback_draft(draft: Mapping[str, Any]) -> bool:
    if bool(draft.get("fallback_generated", False)):
        return True
    if bool(draft.get("degraded_generation", False)):
        return True
    generation_status = _safe_text(draft.get("generation_status")).lower()
    if generation_status in {"fallback_degraded", "fallback_generated"}:
        return True
    provider_status = _safe_text(draft.get("provider_status")).lower()
    return provider_status == "degraded"


def _has_text_generation_degradation(state: Mapping[str, Any]) -> bool:
    drafts = state.get("content_drafts", {})
    if not isinstance(drafts, Mapping):
        return False
    for channel in ("blog", "linkedin"):
        draft = drafts.get(channel, {})
        if isinstance(draft, Mapping) and _is_fallback_draft(draft):
            return True
    return False


def _safe_provider_failure_reason(state: Mapping[str, Any]) -> str:
    drafts = state.get("content_drafts", {})
    if not isinstance(drafts, Mapping):
        return ""
    for channel in ("blog", "linkedin"):
        draft = drafts.get(channel, {})
        if not isinstance(draft, Mapping):
            continue
        reason = _safe_text(draft.get("provider_failure_reason")).lower()
        if reason:
            return reason
    return ""


def _has_recoverable_image_failure(state: Mapping[str, Any]) -> bool:
    errors = state.get("errors", [])
    if isinstance(errors, list):
        for item in errors:
            if not isinstance(item, Mapping):
                continue
            source = _safe_text(item.get("agent") or item.get("node")).lower()
            err_type = _safe_text(item.get("type")).lower()
            recoverable = bool(item.get("recoverable", False))
            if recoverable and (
                "image" in source or "image_generation_failed" in err_type
            ):
                return True
    image_outputs = state.get("image_outputs", [])
    if not isinstance(image_outputs, list):
        return False
    for item in image_outputs:
        if not isinstance(item, Mapping):
            continue
        status = _safe_text(item.get("status")).lower()
        if status in {"failed", "degraded"}:
            return True
        error = item.get("error")
        if isinstance(error, Mapping) and bool(error.get("recoverable", False)):
            return True
    return False


def _has_export_failure(state: Mapping[str, Any]) -> bool:
    summary = _safe_export_outcome_summary(state)
    return int(summary.get("export_error_count", 0)) > 0


def _is_degraded_workflow(state: Mapping[str, Any]) -> bool:
    workflow_status = _safe_text(state.get("workflow_status")).lower()
    if workflow_status in {"partial_success", "completed_with_warnings", "degraded"}:
        return True

    research_data = state.get("research_data", {})
    if isinstance(research_data, Mapping) and bool(
        research_data.get("degraded", False)
    ):
        return True

    if _has_recoverable_image_failure(state):
        return True
    if _has_text_generation_degradation(state):
        return True
    if _has_export_failure(state):
        return True
    return False


def _is_provider_degraded(state: Mapping[str, Any]) -> bool:
    if _is_degraded_workflow(state):
        return True
    research_data = state.get("research_data", {})
    if isinstance(research_data, Mapping) and bool(
        research_data.get("degraded", False)
    ):
        return True
    if _has_text_generation_degradation(state):
        return True
    if _has_recoverable_image_failure(state):
        return True
    return False


def _effective_trace_workflow_status(state: Mapping[str, Any]) -> str:
    workflow_status = _safe_text(state.get("workflow_status")).lower()
    if not workflow_status:
        return ""
    if workflow_status in {"success", "research_complete"} and _is_degraded_workflow(
        state
    ):
        return "partial_success"
    return workflow_status


def _merged_state_view(
    state: Mapping[str, Any],
    updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(state or {})
    if not isinstance(updates, Mapping):
        return merged
    for key, value in updates.items():
        if (
            key in {"retry_counts", "cost_controls", "export_metadata", "research_data"}
            and isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            nested = dict(merged.get(key, {}))
            nested.update(dict(value))
            merged[key] = nested
            continue
        merged[key] = value
    return merged


def safe_trace_metadata(
    state: Mapping[str, Any],
    *,
    node_name: str = "",
    node_status: str = "",
) -> dict[str, Any]:
    """Build safe trace metadata that excludes secrets and raw payloads."""
    requested_outputs = _safe_string_list(state.get("requested_outputs", []))
    routing_decision = _safe_text(state.get("routing_decision"))
    workflow_status = _effective_trace_workflow_status(state)
    session_id = _safe_text(state.get("session_id"))
    retry_metadata = _safe_retry_metadata(state)
    sources_summary = _safe_sources_summary(state)
    image_summary = _safe_image_output_summary(state)
    research_summary = _safe_research_summary(state)
    draft_summary = _safe_draft_summary(state)
    final_response_summary = _safe_final_response_summary(state)
    export_outcome_summary = _safe_export_outcome_summary(state)

    metadata: dict[str, Any] = {
        "requested_outputs": requested_outputs,
        "export_requested": _safe_bool(state.get("export_requested", False)),
        "research_required": _safe_bool(state.get("research_required", False)),
        "clarification_needed": _safe_bool(state.get("clarification_needed", False)),
        "text_generation_degraded": _has_text_generation_degradation(state),
        "image_generation_degraded": _has_recoverable_image_failure(state),
        "fallback_content_used": _has_text_generation_degradation(state),
        "real_generation_succeeded": not _has_text_generation_degradation(state),
        "provider_failure_reason": _safe_provider_failure_reason(state),
        "provider_degraded": _is_provider_degraded(state),
        "degraded_workflow_status": _is_degraded_workflow(state),
        "recoverable_image_failure_status": _has_recoverable_image_failure(state),
        "export_failure_status": _has_export_failure(state),
        "requested_export_formats": export_outcome_summary[
            "requested_export_formats"
        ],
        "completed_export_formats": export_outcome_summary[
            "completed_export_formats"
        ],
        "failed_export_formats": export_outcome_summary["failed_export_formats"],
        "export_warning_count": export_outcome_summary["export_warning_count"],
        "export_error_count": export_outcome_summary["export_error_count"],
        "source_count": _safe_source_count(state),
        "image_output_count": _safe_image_output_count(state),
        "retry_attempt": retry_metadata.get("retry_attempt", 0),
        "retry_exhausted": _safe_bool(retry_metadata.get("retry_exhausted", False)),
        "budget_exceeded": _safe_bool(retry_metadata.get("budget_exceeded", False)),
        "observability_summary": _safe_observability_summary_metadata(),
    }

    if session_id:
        metadata["session_id"] = session_id
    if workflow_status:
        metadata["workflow_status"] = workflow_status
    if routing_decision:
        metadata["routing_decision"] = routing_decision

    retry_summary = _safe_retry_counts_summary(state)
    if retry_summary:
        metadata["retry_count_summary"] = retry_summary

    cost_summary = _safe_cost_controls_summary(state)
    if cost_summary:
        metadata["cost_counter_summary"] = cost_summary

    export_formats_requested = _safe_export_formats(state)
    if export_formats_requested:
        metadata["export_formats_requested"] = export_formats_requested

    error_summary = _safe_error_summary(state)
    if error_summary:
        metadata["error_summary"] = error_summary

    if sources_summary:
        metadata["sources_summary"] = sources_summary
    if image_summary:
        metadata["image_outputs_summary"] = image_summary
    if research_summary:
        metadata["research_summary"] = research_summary
    if draft_summary:
        metadata["content_drafts_summary"] = draft_summary
    if final_response_summary:
        metadata["final_response_summary"] = final_response_summary
    if retry_metadata:
        metadata["retry_metadata"] = retry_metadata

    safe_node_name = _safe_text(node_name)
    if safe_node_name in _AUTHORITATIVE_NODE_SET:
        metadata["node_name"] = safe_node_name

    safe_node_status = _safe_text(node_status).lower()
    if safe_node_status in _STATUS_VALUES:
        metadata["node_status"] = safe_node_status

    sanitized = sanitize_trace_value(metadata)
    sanitized = _strip_unsafe_env_metadata(sanitized)
    if isinstance(sanitized, Mapping):
        return dict(sanitized)
    return {}


def safe_node_start_metadata(
    *,
    state: Mapping[str, Any],
    node_name: str,
) -> dict[str, Any]:
    """Build safe metadata for node start spans."""
    return safe_trace_metadata(state, node_name=node_name, node_status="running")


def safe_node_end_metadata(
    *,
    state: Mapping[str, Any],
    node_name: str,
    node_status: str,
    updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build safe metadata for node end spans."""
    merged_state = _merged_state_view(state, updates)
    return safe_trace_metadata(
        merged_state,
        node_name=node_name,
        node_status=node_status,
    )


def safe_workflow_start_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build safe metadata for workflow start spans."""
    return safe_trace_metadata(state)


def safe_workflow_end_metadata(
    *,
    initial_state: Mapping[str, Any],
    final_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build safe metadata for workflow end spans."""
    return safe_trace_metadata(_merged_state_view(initial_state, final_state))


def safe_tool_metadata(metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build safe metadata for tool-level child spans."""
    candidate = dict(metadata or {})
    safe: dict[str, Any] = {}

    for key in _TOOL_TRACE_STRING_FIELDS:
        value = _safe_text(candidate.get(key))
        if value:
            safe[key] = value

    for key in _TOOL_TRACE_BOOL_FIELDS:
        if key in candidate:
            safe[key] = _safe_bool(candidate.get(key))

    for key in _TOOL_TRACE_INT_FIELDS:
        value = _safe_non_negative_int(candidate.get(key))
        if value is not None:
            safe[key] = value

    safe["observability_summary"] = _safe_observability_summary_metadata()
    sanitized = sanitize_trace_value(safe)
    sanitized = _strip_unsafe_env_metadata(sanitized)
    if isinstance(sanitized, Mapping):
        return dict(sanitized)
    return {}


def start_tool_span(
    tool_name: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> TraceSpanHandle:
    """Start a safe tool-level span using the configured tracer."""
    sampling_context = _SAMPLING_CONTEXT.get()
    if sampling_context is not None and not sampling_context.allow_live_child_spans:
        return _NoOpTraceSpanHandle()

    safe_tool_name = _safe_text(tool_name) or "contentblitz_tool"
    safe_metadata = safe_tool_metadata(
        {
            **dict(metadata or {}),
            "tool_name": safe_tool_name,
        }
    )
    raw_inputs = dict(inputs or {})
    bounded_inputs: dict[str, Any] = {}
    for index, (key, value) in enumerate(raw_inputs.items()):
        if index >= _MAX_TOOL_INPUT_KEYS:
            bounded_inputs["_truncated"] = True
            break
        bounded_inputs[str(key)] = value
    safe_inputs_any = sanitize_trace_value(bounded_inputs)
    safe_inputs = dict(safe_inputs_any) if isinstance(safe_inputs_any, Mapping) else {}
    if "tool_name" not in safe_inputs:
        safe_inputs["tool_name"] = safe_tool_name

    tracer = get_workflow_tracer()
    return tracer.start_tool(
        tool_name=safe_tool_name,
        metadata=safe_metadata,
        inputs=safe_inputs,
    )


class _NoOpTraceSpanHandle:
    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        return None


class _NoOpWorkflowTracer:
    def start_workflow(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        return _NoOpTraceSpanHandle()

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        return _NoOpTraceSpanHandle()

    def start_tool(
        self,
        *,
        tool_name: str,
        metadata: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
    ) -> TraceSpanHandle:
        return _NoOpTraceSpanHandle()


class _SafeTraceSpanHandle:
    """Guard trace span finish from raising into workflow execution."""

    def __init__(self, delegate: TraceSpanHandle | None) -> None:
        self._delegate = delegate

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self._delegate is None:
            return
        try:
            self._delegate.finish(
                metadata=metadata,
                outputs=outputs,
                error=error,
            )
        except Exception:
            return None


class _SafeWorkflowTracer:
    """Guard tracer start methods from raising into workflow execution."""

    def __init__(
        self,
        delegate: WorkflowTracer,
        *,
        config: ObservabilityConfig,
    ) -> None:
        self._delegate = delegate
        self._config = config

    def start_workflow(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        safe_metadata_any = sanitize_trace_value(dict(metadata or {}))
        safe_metadata = (
            dict(safe_metadata_any)
            if isinstance(safe_metadata_any, Mapping)
            else {}
        )
        stripped_metadata_any = _strip_unsafe_env_metadata(safe_metadata)
        safe_metadata = (
            dict(stripped_metadata_any)
            if isinstance(stripped_metadata_any, Mapping)
            else {}
        )
        decision = _build_sampling_decision(
            metadata=safe_metadata,
            config=self._config,
        )
        allow_live_child_spans = decision.success_sampled or decision.failure_sampled
        context_token = _SAMPLING_CONTEXT.set(
            _TraceSamplingContext(allow_live_child_spans=allow_live_child_spans)
        )
        if not allow_live_child_spans:
            return _SamplingAwareTraceSpanHandle(
                delegate_tracer=self._delegate,
                start_metadata=safe_metadata,
                decision=decision,
                context_token=context_token,
            )

        # When both sample rates are 1.0 we can eagerly start a span while still
        # retaining context reset guarantees through the sampling-aware wrapper.
        eager_delegate_handle: TraceSpanHandle | None = None
        if (
            self._config.trace_sample_rate >= 1.0
            and self._config.trace_failure_sample_rate >= 1.0
        ):
            try:
                eager_delegate_handle = self._delegate.start_workflow(
                    metadata=safe_metadata
                )
            except Exception:
                _SAMPLING_CONTEXT.reset(context_token)
                return _NoOpTraceSpanHandle()

        return _SamplingAwareTraceSpanHandle(
            delegate_tracer=self._delegate,
            start_metadata=safe_metadata,
            decision=decision,
            context_token=context_token,
            delegate_handle=eager_delegate_handle,
        )

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        sampling_context = _SAMPLING_CONTEXT.get()
        if sampling_context is not None and not sampling_context.allow_live_child_spans:
            return _NoOpTraceSpanHandle()
        try:
            handle = self._delegate.start_node(
                node_name=node_name,
                metadata=metadata,
            )
        except Exception:
            return _NoOpTraceSpanHandle()
        return _SafeTraceSpanHandle(handle)

    def start_tool(
        self,
        *,
        tool_name: str,
        metadata: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
    ) -> TraceSpanHandle:
        sampling_context = _SAMPLING_CONTEXT.get()
        if sampling_context is not None and not sampling_context.allow_live_child_spans:
            return _NoOpTraceSpanHandle()
        start_tool_method = getattr(self._delegate, "start_tool", None)
        if not callable(start_tool_method):
            return _NoOpTraceSpanHandle()
        try:
            handle = start_tool_method(
                tool_name=tool_name,
                metadata=metadata,
                inputs=inputs,
            )
        except Exception:
            return _NoOpTraceSpanHandle()
        return _SafeTraceSpanHandle(handle)


class _LangSmithTraceSpanHandle:
    def __init__(self, trace_context_manager: Any) -> None:
        self._trace_cm = trace_context_manager
        self._run: Any = None
        self._closed = False
        try:
            self._run = self._trace_cm.__enter__()
        except Exception:
            self._closed = True
            self._run = None

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if self._closed:
            return

        safe_outputs_any = sanitize_trace_value(dict(outputs or {}))
        safe_outputs = (
            dict(safe_outputs_any)
            if isinstance(safe_outputs_any, Mapping)
            else {}
        )
        stripped_outputs_any = _strip_unsafe_env_metadata(safe_outputs)
        safe_outputs = (
            dict(stripped_outputs_any)
            if isinstance(stripped_outputs_any, Mapping)
            else {}
        )
        safe_metadata_any = sanitize_trace_value(dict(metadata or {}))
        safe_metadata = (
            dict(safe_metadata_any)
            if isinstance(safe_metadata_any, Mapping)
            else {}
        )
        stripped_metadata_any = _strip_unsafe_env_metadata(safe_metadata)
        safe_metadata = (
            dict(stripped_metadata_any)
            if isinstance(stripped_metadata_any, Mapping)
            else {}
        )
        safe_workflow_inputs = safe_workflow_trace_inputs(safe_metadata)
        metadata_workflow_status = _safe_text(safe_metadata.get("workflow_status"))
        if metadata_workflow_status:
            safe_outputs["workflow_status"] = metadata_workflow_status

        def _scrub_run_metadata() -> None:
            if self._run is None:
                return
            metadata_attr = getattr(self._run, "metadata", None)
            if isinstance(metadata_attr, dict):
                cleaned_existing = _strip_unsafe_env_metadata(dict(metadata_attr))
                metadata_attr.clear()
                metadata_attr.update(
                    cleaned_existing if isinstance(cleaned_existing, Mapping) else {}
                )
                metadata_attr.update(safe_metadata)

            extra_attr = getattr(self._run, "extra", None)
            if isinstance(extra_attr, dict):
                extra_metadata = extra_attr.get("metadata")
                if isinstance(extra_metadata, dict):
                    cleaned_extra = _strip_unsafe_env_metadata(dict(extra_metadata))
                    extra_metadata.clear()
                    extra_metadata.update(
                        cleaned_extra if isinstance(cleaned_extra, Mapping) else {}
                    )
                    extra_metadata.update(safe_metadata)

        def _scrub_run_inputs() -> None:
            if self._run is None or not safe_workflow_inputs:
                return
            existing_inputs = getattr(self._run, "inputs", None)
            if isinstance(existing_inputs, dict):
                existing_inputs.clear()
                existing_inputs.update(safe_workflow_inputs)
                return
            try:
                setattr(self._run, "inputs", dict(safe_workflow_inputs))
            except Exception:
                return

        try:
            _scrub_run_metadata()
            _scrub_run_inputs()
        except Exception:
            # Tracing must never fail workflow execution.
            pass

        try:
            if self._run is not None and hasattr(self._run, "end"):
                if error is not None:
                    safe_error = normalize_trace_error(error)
                    self._run.end(
                        outputs=safe_outputs,
                        error=str(safe_error.get("code", "workflow_error")),
                    )
                else:
                    self._run.end(outputs=safe_outputs)
        except Exception:
            pass
        finally:
            try:
                _scrub_run_metadata()
                _scrub_run_inputs()
            except Exception:
                pass
            try:
                self._trace_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._closed = True


class _LangSmithWorkflowTracer:
    def __init__(self, *, project: str, endpoint: str) -> None:
        langsmith_module = import_module("langsmith")
        run_helpers = import_module("langsmith.run_helpers")

        client_cls = getattr(langsmith_module, "Client")
        self._trace_ctor = getattr(run_helpers, "trace")
        self._project = project
        # API key is loaded by LangSmith client from environment; never expose it.
        self._client = client_cls(
            api_url=endpoint,
            omit_traced_runtime_info=True,
        )

    def start_workflow(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        safe_metadata_any = sanitize_trace_value(dict(metadata or {}))
        safe_metadata = (
            dict(safe_metadata_any)
            if isinstance(safe_metadata_any, Mapping)
            else {}
        )
        stripped_metadata_any = _strip_unsafe_env_metadata(safe_metadata)
        safe_metadata = (
            dict(stripped_metadata_any)
            if isinstance(stripped_metadata_any, Mapping)
            else {}
        )
        trace_ctx = self._trace_ctor(
            "contentblitz_workflow",
            run_type="chain",
            inputs=safe_workflow_trace_inputs(safe_metadata),
            metadata=safe_metadata,
            project_name=self._project,
            client=self._client,
        )
        return _LangSmithTraceSpanHandle(trace_ctx)

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        # LangGraph's native LangSmith integration already emits node spans.
        # Returning a no-op here avoids duplicate node spans with identical names.
        return _NoOpTraceSpanHandle()

    def start_tool(
        self,
        *,
        tool_name: str,
        metadata: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
    ) -> TraceSpanHandle:
        safe_metadata = safe_tool_metadata(metadata)
        safe_tool_name = (
            _safe_text(safe_metadata.get("tool_name"))
            or _safe_text(tool_name)
            or "contentblitz_tool"
        )
        safe_inputs_any = sanitize_trace_value(dict(inputs or {}))
        safe_inputs = (
            dict(safe_inputs_any) if isinstance(safe_inputs_any, Mapping) else {}
        )
        if "tool_name" not in safe_inputs:
            safe_inputs["tool_name"] = safe_tool_name
        trace_ctx = self._trace_ctor(
            safe_tool_name,
            run_type="tool",
            inputs=safe_inputs,
            metadata=safe_metadata,
            project_name=self._project,
            client=self._client,
        )
        return _LangSmithTraceSpanHandle(trace_ctx)


def build_observability_config() -> ObservabilityConfig:
    """Build secret-safe, import-time-safe observability settings."""
    tracing_requested = langsmith_tracing_requested()
    has_api_key = langsmith_api_key_present()
    tracing_enabled = bool(tracing_requested and has_api_key)
    trace_sample_rate = _read_sample_rate_env(
        _TRACE_SAMPLE_RATE_ENV,
        default=_DEFAULT_TRACE_SAMPLE_RATE,
    )
    trace_failure_sample_rate = _read_sample_rate_env(
        _TRACE_FAILURE_SAMPLE_RATE_ENV,
        default=_DEFAULT_FAILURE_TRACE_SAMPLE_RATE,
    )

    if tracing_enabled:
        status = "enabled"
        message = "LangSmith tracing is enabled."
    elif tracing_requested:
        status = "degraded"
        message = (
            "LangSmith tracing was requested but LANGSMITH_API_KEY is missing. "
            "Tracing remains disabled."
        )
    else:
        status = "disabled"
        message = "LangSmith tracing is disabled."

    return ObservabilityConfig(
        tracing_requested=tracing_requested,
        tracing_enabled=tracing_enabled,
        trace_sample_rate=trace_sample_rate,
        trace_failure_sample_rate=trace_failure_sample_rate,
        endpoint=langsmith_endpoint(),
        project=langsmith_project(),
        status=status,
        message=message,
    )


def is_tracing_enabled() -> bool:
    """Return whether tracing is currently active."""
    return build_observability_config().tracing_enabled


def observability_summary() -> Dict[str, str | bool | float]:
    """Return a secret-safe observability snapshot for UI/logging/debug."""
    config = build_observability_config()
    return {
        "tracing_requested": config.tracing_requested,
        "tracing_enabled": config.tracing_enabled,
        "trace_sample_rate": config.trace_sample_rate,
        "trace_failure_sample_rate": config.trace_failure_sample_rate,
        "endpoint": config.endpoint,
        "project": config.project,
        "status": config.status,
        "message": config.message,
    }


_TracerFactory = Callable[[ObservabilityConfig], WorkflowTracer]
_tracer_factory_override: _TracerFactory | None = None


def _default_tracer_factory(config: ObservabilityConfig) -> WorkflowTracer:
    if not config.tracing_enabled:
        return _NoOpWorkflowTracer()
    try:
        return _LangSmithWorkflowTracer(
            project=config.project,
            endpoint=config.endpoint,
        )
    except Exception:
        # Tracing setup failure must degrade safely.
        return _NoOpWorkflowTracer()


def set_tracer_factory(factory: _TracerFactory) -> None:
    """Override tracer factory for tests."""
    global _tracer_factory_override
    _tracer_factory_override = factory


def reset_tracer_factory() -> None:
    """Reset tracer factory override."""
    global _tracer_factory_override
    _tracer_factory_override = None


def get_workflow_tracer() -> WorkflowTracer:
    """Return the configured workflow tracer."""
    config = build_observability_config()
    factory = _tracer_factory_override or _default_tracer_factory
    try:
        tracer = factory(config)
    except Exception:
        tracer = _NoOpWorkflowTracer()
    return _SafeWorkflowTracer(tracer, config=config)
