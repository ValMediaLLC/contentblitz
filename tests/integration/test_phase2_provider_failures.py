import importlib

generate_text_module = importlib.import_module("contentblitz.tools.generate_text")
generate_text = generate_text_module.generate_text


class FailingResponses:
    def create(self, **kwargs):
        raise RuntimeError("raw provider exploded with sk-test-secret-value")


class FailingOpenAI:
    def __init__(self, *args, **kwargs):
        self.responses = FailingResponses()


def test_generate_text_provider_failure_is_normalized(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(generate_text_module, "OpenAI", FailingOpenAI)

    result = generate_text(
        prompt="This should fail safely.",
        agent_key="query_handler",
        model="gpt-4o-mini",
    )

    assert result.degraded is True
    assert result.text == ""
    assert result.error is not None

    error_text = str(result.error)

    assert "sk-test-secret-value" not in error_text
    assert "Traceback" not in error_text
    assert "raw provider exploded" not in error_text


def test_generate_text_invalid_agent_key_fails_safely():
    result = generate_text(
        prompt="Invalid agent key test.",
        agent_key="not_a_real_agent",
        model="gpt-4o-mini",
    )

    assert result.degraded is True
    assert result.error is not None
    assert "Traceback" not in str(result.error)