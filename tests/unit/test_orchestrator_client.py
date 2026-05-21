from __future__ import annotations

from frontend.services import orchestrator_client as orchestrator_client_module


def test_blog_writer_provider_prefers_provider_used() -> None:
    provider, model = orchestrator_client_module._node_provider_and_model(  # noqa: SLF001
        node_name="blog_writer_node",
        updates={
            "content_drafts": {
                "blog": {
                    "provider_used": "anthropic",
                    "model_used": "claude-sonnet-4-6",
                }
            }
        },
    )

    assert provider == "anthropic"
    assert model == "claude-sonnet-4-6"


def test_blog_writer_provider_infers_anthropic_from_claude_model() -> None:
    provider, model = orchestrator_client_module._node_provider_and_model(  # noqa: SLF001
        node_name="blog_writer_node",
        updates={"content_drafts": {"blog": {"model_used": "claude-sonnet-4-6"}}},
    )

    assert provider == "anthropic"
    assert model == "claude-sonnet-4-6"


def test_linkedin_writer_provider_infers_openai_from_gpt_model() -> None:
    provider, model = orchestrator_client_module._node_provider_and_model(  # noqa: SLF001
        node_name="linkedin_writer_node",
        updates={"content_drafts": {"linkedin": {"model_used": "gpt-5.4"}}},
    )

    assert provider == "openai"
    assert model == "gpt-5.4"


def test_image_agent_model_reads_model_field_not_provider() -> None:
    provider, model = orchestrator_client_module._node_provider_and_model(  # noqa: SLF001
        node_name="image_agent_node",
        updates={
            "tool_outputs": {
                "image_agent": {
                    "provider": "dall-e-3",
                    "model": "gpt-image-1",
                }
            }
        },
    )

    assert provider == "dall-e-3"
    assert model == "gpt-image-1"
