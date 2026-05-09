import importlib
from types import SimpleNamespace

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
generate_text = generate_text_module.generate_text


class FakeResponse:
    output_text = "ContentBlitz contract test passed."
    usage = SimpleNamespace(
        input_tokens=5,
        output_tokens=7,
        total_tokens=12,
        prompt_tokens=5,
        completion_tokens=7,
    )

    choices = [
        SimpleNamespace(
            message=SimpleNamespace(content="ContentBlitz contract test passed.")
        )
    ]


class FakeResponses:
    def create(self, *args, **kwargs):
        return FakeResponse()


class FakeChatCompletions:
    def create(self, *args, **kwargs):
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeChatCompletions()


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.responses = FakeResponses()
        self.chat = FakeChat()


def test_generate_text_returns_normalized_result_without_state(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(generate_text_module, "OpenAI", FakeOpenAI)

    result = generate_text(
        prompt="Say ContentBlitz contract test passed.",
        agent_key="query_handler",
        model="gpt-4o-mini",
    )

    assert result.text
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"
    assert result.degraded is False
    assert result.error is None
    assert result.total_tokens >= 0


def test_generate_text_tool_does_not_require_state(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(generate_text_module, "OpenAI", FakeOpenAI)

    result = generate_text(
        prompt="Say no state needed.",
        agent_key="query_handler",
        model="gpt-4o-mini",
    )

    assert result.text
    assert result.degraded is False