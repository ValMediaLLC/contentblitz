from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from contentblitz.core import observability as observability_module
from contentblitz.tools.provider_types import SearchResult, SearchWebResult

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
search_web_module = importlib.import_module("contentblitz.tools.search_web")
generate_image_module = importlib.import_module("contentblitz.tools.generate_image")
cache_module = importlib.import_module("contentblitz.tools.cache")


def _collect_metadata_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_collect_metadata_keys(nested))
        return keys
    if isinstance(value, list):
        for nested in value:
            keys.update(_collect_metadata_keys(nested))
    return keys


def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)


@dataclass
class _RecordingSpan:
    sink: list[dict[str, Any]]
    label: str

    def finish(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.sink.append(
            {
                "event": "tool_finish",
                "tool_name": self.label,
                "metadata": dict(metadata or {}),
                "outputs": dict(outputs or {}),
                "error": error.__class__.__name__ if error else "",
            }
        )


@dataclass
class _RecordingTracer:
    events: list[dict[str, Any]] = field(default_factory=list)

    def start_workflow(self, *, metadata: Mapping[str, Any]) -> Any:
        return _RecordingSpan(self.events, "workflow")

    def start_node(self, *, node_name: str, metadata: Mapping[str, Any]) -> Any:
        return _RecordingSpan(self.events, node_name)

    def start_tool(
        self,
        *,
        tool_name: str,
        metadata: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        self.events.append(
            {
                "event": "tool_start",
                "tool_name": str(tool_name),
                "metadata": dict(metadata),
                "inputs": dict(inputs or {}),
            }
        )
        return _RecordingSpan(self.events, str(tool_name))


@dataclass
class _BrokenToolTracer:
    def start_workflow(self, *, metadata: Mapping[str, Any]) -> Any:
        return _RecordingSpan([], "workflow")

    def start_node(self, *, node_name: str, metadata: Mapping[str, Any]) -> Any:
        return _RecordingSpan([], node_name)

    def start_tool(
        self,
        *,
        tool_name: str,
        metadata: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        raise RuntimeError("tool tracer failed")


@pytest.fixture(autouse=True)
def _reset_tracing_factory() -> None:
    observability_module.reset_tracer_factory()
    yield
    observability_module.reset_tracer_factory()


def _install_text_client(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    response = SimpleNamespace(
        model="gpt-4o",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20),
    )
    completions = SimpleNamespace(create=lambda **_kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: client,
    )


def _install_image_client_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _generate(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("primary failed")
        return SimpleNamespace(
            data=[SimpleNamespace(url="https://img.example/fallback.png")]
        )

    client = SimpleNamespace(images=SimpleNamespace(generate=_generate))
    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: client,
    )


def test_generate_text_creates_child_span_when_tracing_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-secret")
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    _install_text_client(monkeypatch, "safe output text")

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)

    result = generate_text_module.generate_text(
        prompt="raw prompt with sk-live-secret that must not be traced",
        agent_key="blog_writer",
    )

    assert result.degraded is False
    tool_starts = [
        event for event in recording.events if event.get("event") == "tool_start"
    ]
    assert any(event.get("tool_name") == "generate_text" for event in tool_starts)
    tool_finish = next(
        event
        for event in recording.events
        if event.get("event") == "tool_finish"
        and event.get("tool_name") == "generate_text"
    )
    finish_metadata = tool_finish["metadata"]
    assert finish_metadata["provider"] == "openai"
    assert finish_metadata["model"] == "gpt-4o"
    assert finish_metadata["total_token_count"] == 20
    assert finish_metadata["retry_attempt"] >= 1
    assert finish_metadata["fallback_used"] is False
    assert finish_metadata["observability_summary"]["provider"] == "langsmith"
    assert "endpoint_host" in finish_metadata["observability_summary"]
    all_keys = _collect_metadata_keys(tool_finish)
    assert "LANGSMITH_TRACING" not in all_keys
    assert "LANGSMITH_ENDPOINT" not in all_keys
    assert "LANGSMITH_PROJECT" not in all_keys
    assert "LANGSMITH_API_KEY" not in all_keys
    assert not any(key.upper().endswith("_API_KEY") for key in all_keys)
    flattened = repr(recording.events).lower()
    assert "sk-live-secret" not in flattened
    assert "raw prompt" not in flattened
    assert "safe output text" not in flattened


def test_search_web_fallback_to_perplexity_is_traced_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")

    monkeypatch.setattr(
        search_web_module,
        "_search_serp",
        lambda query, max_results: SearchWebResult(
            provider="serp",
            query=query,
            results=[],
            degraded=True,
            error={"code": "provider_error"},
        ),
    )
    monkeypatch.setattr(
        search_web_module,
        "_search_perplexity_placeholder",
        lambda query, max_results: SearchWebResult(
            provider="perplexity",
            query=query,
            results=[
                SearchResult(
                    title="Fallback source",
                    url=None,
                    snippet="Perplexity fallback snippet for testing.",
                    source="perplexity",
                    published_at=None,
                    citation_available=False,
                    credibility_score=0.35,
                )
            ],
            degraded=False,
            error=None,
        ),
    )

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)
    result = search_web_module.search_web("trace fallback", provider="auto")

    assert result.provider == "perplexity"
    tool_names = [
        event.get("tool_name")
        for event in recording.events
        if event.get("event") == "tool_start"
    ]
    assert "search_web" in tool_names
    assert "serp" in tool_names
    assert "perplexity_fallback" in tool_names
    search_finish = next(
        event
        for event in recording.events
        if event.get("event") == "tool_finish"
        and event.get("tool_name") == "search_web"
    )
    search_metadata = search_finish["metadata"]
    assert search_metadata["fallback_used"] is True
    assert search_metadata["fallback_provider"] == "perplexity"
    assert search_metadata["result_count"] == 1
    assert search_metadata["source_count"] == 1
    flattened = repr(recording.events).lower()
    assert "trace fallback" not in flattened


def test_generate_image_fallback_span_and_no_base64_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-secret")
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    _install_image_client_with_fallback(monkeypatch)

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)
    result = generate_image_module.generate_image(
        prompt="image prompt with base64 token"
    )

    assert result.degraded is False
    assert result.model == "dall-e-2"
    tool_names = [
        event.get("tool_name")
        for event in recording.events
        if event.get("event") == "tool_start"
    ]
    assert "generate_image" in tool_names
    assert "dall_e_3" in tool_names
    assert "dall_e_2_fallback" in tool_names
    image_finish = next(
        event
        for event in recording.events
        if event.get("event") == "tool_finish"
        and event.get("tool_name") == "generate_image"
    )
    image_metadata = image_finish["metadata"]
    assert image_metadata["fallback_used"] is True
    assert image_metadata["fallback_model"] == "dall-e-2"
    assert image_metadata["final_model"] == "dall-e-2"
    assert image_metadata["image_output_count"] == 1
    flattened = repr(recording.events).lower()
    assert "base64" not in flattened
    assert "image prompt" not in flattened


def test_tracing_disabled_uses_noop_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_langsmith_env(monkeypatch)
    config = observability_module.build_observability_config()
    tracer = observability_module._default_tracer_factory(config)  # noqa: SLF001
    assert config.tracing_enabled is False
    assert tracer.__class__.__name__ == "_NoOpWorkflowTracer"


def test_cache_lookup_and_write_spans_are_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    cache_module.clear_cache()

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)
    key = cache_module.build_research_cache_key("observability cache test")

    assert cache_module.set_cache(key, {"safe": "value"}, ttl_seconds=30) is True
    assert cache_module.get_cache(key) == {"safe": "value"}

    tool_names = [
        event.get("tool_name")
        for event in recording.events
        if event.get("event") == "tool_start"
    ]
    assert "cache_write" in tool_names
    assert "cache_lookup" in tool_names
    write_finish = next(
        event
        for event in recording.events
        if event.get("event") == "tool_finish"
        and event.get("tool_name") == "cache_write"
    )
    lookup_finish = next(
        event
        for event in recording.events
        if event.get("event") == "tool_finish"
        and event.get("tool_name") == "cache_lookup"
    )
    assert write_finish["metadata"]["cache_hit"] is False
    assert write_finish["metadata"]["cache_miss"] is True
    assert lookup_finish["metadata"]["cache_hit"] is True
    assert lookup_finish["metadata"]["cache_miss"] is False
    cache_module.clear_cache()


def test_tool_tracing_failure_does_not_fail_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    _install_text_client(monkeypatch, "tool still works")

    observability_module.set_tracer_factory(lambda _config: _BrokenToolTracer())
    result = generate_text_module.generate_text(
        prompt="normal prompt",
        agent_key="query_handler",
    )
    assert result.degraded is False
    assert result.text == "tool still works"


def test_langsmith_node_spans_are_noop_to_avoid_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_calls: list[str] = []
    client_init_kwargs: list[dict[str, Any]] = []
    run_holder: dict[str, Any] = {}

    class _FakeClient:
        def __init__(
            self,
            *,
            api_url: str,
            omit_traced_runtime_info: bool = False,
        ) -> None:
            client_init_kwargs.append(
                {
                    "api_url": api_url,
                    "omit_traced_runtime_info": omit_traced_runtime_info,
                }
            )
            self.api_url = api_url

    class _FakeTraceContext:
        def __enter__(self) -> Any:
            run = SimpleNamespace(
                metadata={
                    "LANGSMITH_TRACING": "true",
                    "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
                    "LANGSMITH_PROJECT": "ContentBlitz",
                },
                extra={
                    "metadata": {
                        "LANGSMITH_TRACING": "true",
                        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
                        "LANGSMITH_PROJECT": "ContentBlitz",
                    }
                },
                end=lambda **_kwargs: None,
            )
            run_holder["run"] = run
            return run

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

    def _trace(name: str, **_kwargs: Any) -> Any:
        trace_calls.append(name)
        return _FakeTraceContext()

    fake_langsmith = SimpleNamespace(Client=_FakeClient)
    fake_run_helpers = SimpleNamespace(trace=_trace)

    def _fake_import(name: str) -> Any:
        if name == "langsmith":
            return fake_langsmith
        if name == "langsmith.run_helpers":
            return fake_run_helpers
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(observability_module, "import_module", _fake_import)
    tracer = observability_module._LangSmithWorkflowTracer(  # noqa: SLF001
        project="ContentBlitz",
        endpoint="https://api.smith.langchain.com",
    )
    assert client_init_kwargs == [
        {
            "api_url": "https://api.smith.langchain.com",
            "omit_traced_runtime_info": True,
        }
    ]

    node_span = tracer.start_node(
        node_name="blog_writer_node",
        metadata={"node_name": "blog_writer_node"},
    )
    node_span.finish(outputs={"status": "completed"})
    assert trace_calls == []

    tool_span = tracer.start_tool(
        tool_name="generate_text",
        metadata={"tool_name": "generate_text", "provider": "openai"},
        inputs={"tool_name": "generate_text"},
    )
    tool_span.finish(outputs={"degraded": False})
    assert trace_calls == ["generate_text"]
    run = run_holder["run"]
    assert "LANGSMITH_TRACING" not in run.metadata
    assert "LANGSMITH_ENDPOINT" not in run.metadata
    assert "LANGSMITH_PROJECT" not in run.metadata
    assert "LANGSMITH_TRACING" not in run.extra["metadata"]
    assert "LANGSMITH_ENDPOINT" not in run.extra["metadata"]
    assert "LANGSMITH_PROJECT" not in run.extra["metadata"]
