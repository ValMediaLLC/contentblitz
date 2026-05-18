"""Observability configuration and safe tracing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Protocol

from contentblitz.config import (
    langsmith_api_key_present,
    langsmith_endpoint,
    langsmith_project,
    langsmith_tracing_requested,
)

_STATUS_VALUES = {"pending", "running", "completed", "degraded", "failed", "skipped"}
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


@dataclass(frozen=True)
class ObservabilityConfig:
    """Public, secret-safe observability configuration."""

    tracing_requested: bool
    tracing_enabled: bool
    endpoint: str
    project: str
    status: str
    message: str


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


def _safe_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or text.lower() in {"none", "null"}:
        return ""
    return text


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
    for key in _COST_COUNTER_KEYS:
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
    return _safe_string_list(export_metadata.get("formats_requested", []))


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
    export_metadata = state.get("export_metadata", {})
    if not isinstance(export_metadata, Mapping):
        return False
    error_log = export_metadata.get("error_log", [])
    if isinstance(error_log, list) and any(
        isinstance(item, Mapping) for item in error_log
    ):
        return True
    export_status = export_metadata.get("export_status", {})
    if isinstance(export_status, Mapping):
        for value in export_status.values():
            if _safe_text(value).lower() == "failed":
                return True
    return False


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
    if _has_export_failure(state):
        return True
    return False


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
    workflow_status = _safe_text(state.get("workflow_status"))
    session_id = _safe_text(state.get("session_id"))

    metadata: dict[str, Any] = {
        "requested_outputs": requested_outputs,
        "export_requested": _safe_bool(state.get("export_requested", False)),
        "research_required": _safe_bool(state.get("research_required", False)),
        "clarification_needed": _safe_bool(state.get("clarification_needed", False)),
        "degraded_workflow_status": _is_degraded_workflow(state),
        "recoverable_image_failure_status": _has_recoverable_image_failure(state),
        "export_failure_status": _has_export_failure(state),
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

    safe_node_name = _safe_text(node_name)
    if safe_node_name in _AUTHORITATIVE_NODE_SET:
        metadata["node_name"] = safe_node_name

    safe_node_status = _safe_text(node_status).lower()
    if safe_node_status in _STATUS_VALUES:
        metadata["node_status"] = safe_node_status

    return metadata


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

    def __init__(self, delegate: WorkflowTracer) -> None:
        self._delegate = delegate

    def start_workflow(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        try:
            handle = self._delegate.start_workflow(metadata=metadata)
        except Exception:
            return _NoOpTraceSpanHandle()
        return _SafeTraceSpanHandle(handle)

    def start_node(
        self,
        *,
        node_name: str,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        try:
            handle = self._delegate.start_node(
                node_name=node_name,
                metadata=metadata,
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

        safe_outputs = dict(outputs or {})
        safe_metadata = dict(metadata or {})

        try:
            metadata_attr = getattr(self._run, "metadata", None)
            if self._run is not None and isinstance(metadata_attr, dict):
                self._run.metadata.update(safe_metadata)
        except Exception:
            # Tracing must never fail workflow execution.
            pass

        try:
            if self._run is not None and hasattr(self._run, "end"):
                if error is not None:
                    self._run.end(
                        outputs=safe_outputs,
                        error=error.__class__.__name__,
                    )
                else:
                    self._run.end(outputs=safe_outputs)
        except Exception:
            pass
        finally:
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
        self._client = client_cls(api_url=endpoint)

    def start_workflow(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> TraceSpanHandle:
        trace_ctx = self._trace_ctor(
            "contentblitz_workflow",
            run_type="chain",
            inputs={"requested_outputs": list(metadata.get("requested_outputs", []))},
            metadata=dict(metadata),
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
        trace_ctx = self._trace_ctor(
            node_name,
            run_type="tool",
            inputs={"node_name": node_name},
            metadata=dict(metadata),
            project_name=self._project,
            client=self._client,
        )
        return _LangSmithTraceSpanHandle(trace_ctx)


def build_observability_config() -> ObservabilityConfig:
    """Build secret-safe, import-time-safe observability settings."""
    tracing_requested = langsmith_tracing_requested()
    has_api_key = langsmith_api_key_present()
    tracing_enabled = bool(tracing_requested and has_api_key)

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
        endpoint=langsmith_endpoint(),
        project=langsmith_project(),
        status=status,
        message=message,
    )


def is_tracing_enabled() -> bool:
    """Return whether tracing is currently active."""
    return build_observability_config().tracing_enabled


def observability_summary() -> Dict[str, str | bool]:
    """Return a secret-safe observability snapshot for UI/logging/debug."""
    config = build_observability_config()
    return {
        "tracing_requested": config.tracing_requested,
        "tracing_enabled": config.tracing_enabled,
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
    return _SafeWorkflowTracer(tracer)
