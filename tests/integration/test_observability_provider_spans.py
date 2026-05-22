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
                "event": "finish",
                "name": self.label,
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
                "event": "start",
                "name": str(tool_name),
                "metadata": dict(metadata),
                "inputs": dict(inputs or {}),
            }
        )
        return _RecordingSpan(self.events, str(tool_name))


@pytest.fixture(autouse=True)
def _reset_tracing_factory() -> None:
    observability_module.reset_tracer_factory()
    yield
    observability_module.reset_tracer_factory()


def _install_mocked_text_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        model="gpt-4o",
        choices=[SimpleNamespace(message=SimpleNamespace(content="safe content"))],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=5, total_tokens=12),
    )
    completions = SimpleNamespace(create=lambda **_kwargs: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        generate_text_module,
        "_build_openai_client",
        lambda api_key: client,
    )


def _install_mocked_image_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _generate(**kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("primary failed")
        return SimpleNamespace(data=[SimpleNamespace(url="https://img.example/a.png")])

    client = SimpleNamespace(images=SimpleNamespace(generate=_generate))
    monkeypatch.setattr(
        generate_image_module,
        "_build_openai_client",
        lambda api_key: client,
    )
    monkeypatch.setattr(
        generate_image_module,
        "_build_fal_client",
        lambda api_key: client,
    )


def test_provider_child_spans_are_recorded_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-secret")

    _install_mocked_text_provider(monkeypatch)
    _install_mocked_image_provider(monkeypatch)
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
                    title="Perplexity",
                    url="https://example.com/source",
                    snippet="usable fallback snippet",
                    source="perplexity",
                    published_at=None,
                    citation_available=True,
                    credibility_score=0.35,
                )
            ],
            degraded=False,
            error=None,
        ),
    )

    recording = _RecordingTracer()
    observability_module.set_tracer_factory(lambda _config: recording)

    _ = generate_text_module.generate_text(
        prompt="prompt text must not leak",
        agent_key="blog_writer",
    )
    _ = search_web_module.search_web("search text must not leak", provider="auto")
    _ = generate_image_module.generate_image(prompt="base64-like prompt")

    tool_start_names = [
        event["name"] for event in recording.events if event.get("event") == "start"
    ]
    assert "generate_text" in tool_start_names
    assert "search_web" in tool_start_names
    assert "serp" in tool_start_names
    assert "perplexity_fallback" in tool_start_names
    assert "generate_image" in tool_start_names
    assert "stability_ai" in tool_start_names
    assert "fal_ai_fallback" in tool_start_names

    tool_finishes = {
        event["name"]: event
        for event in recording.events
        if event.get("event") == "finish"
    }
    assert tool_finishes["generate_text"]["metadata"]["total_token_count"] == 12
    assert tool_finishes["search_web"]["metadata"]["fallback_used"] is True
    assert tool_finishes["search_web"]["metadata"]["fallback_provider"] == "perplexity"
    assert tool_finishes["generate_image"]["metadata"]["fallback_used"] is True
    assert (
        tool_finishes["generate_image"]["metadata"]["final_model"]
        == "fal-ai/fast-sdxl"
    )
    assert (
        tool_finishes["perplexity_fallback"]["metadata"]["observability_summary"][
            "provider"
        ]
        == "langsmith"
    )

    flattened = repr(recording.events).lower()
    serialized = str(recording.events)
    metadata_keys = _collect_metadata_keys(recording.events)
    assert "sk-live-secret" not in flattened
    assert "prompt text must not leak" not in flattened
    assert "search text must not leak" not in flattened
    assert "base64" not in flattened
    assert "LANGSMITH_TRACING" not in metadata_keys
    assert "LANGSMITH_ENDPOINT" not in metadata_keys
    assert "LANGSMITH_PROJECT" not in metadata_keys
    assert "LANGSMITH_API_KEY" not in metadata_keys
    assert not any(key.upper().endswith("_API_KEY") for key in metadata_keys)
    assert "https://api.smith.langchain.com" not in serialized


def test_tools_execute_without_langsmith_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_langsmith_env(monkeypatch)
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERP_API_KEY", raising=False)

    text_result = generate_text_module.generate_text(
        prompt="no tracing credentials needed",
        agent_key="query_handler",
    )
    search_result = search_web_module.search_web("safe query", provider="serp")
    image_result = generate_image_module.generate_image(prompt="safe image prompt")

    assert text_result.degraded is True
    assert search_result.degraded is True
    assert image_result.degraded is True
