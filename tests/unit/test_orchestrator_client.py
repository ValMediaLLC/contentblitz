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


def test_image_agent_event_metadata_includes_provider_attempt_chain() -> None:
    metadata = orchestrator_client_module._event_metadata(  # noqa: SLF001
        {
            "workflow_status": "partial_success",
            "tool_outputs": {
                "image_agent": {
                    "provider": "fal_ai",
                    "model": "fal-ai/flux/schnell",
                    "provider_call_count": 2,
                    "provider_call_count_by_provider": {
                        "stability_ai": 1,
                        "fal_ai": 1,
                    },
                    "provider_latency_by_provider_ms": {
                        "stability_ai": 420,
                        "fal_ai": 510,
                    },
                    "image_provider_attempts": [
                        {
                            "provider": "stability_ai",
                            "model": "stable-image-core",
                            "status": "failed",
                            "error_code": "authentication_failed",
                            "duration_ms": 420,
                            "fallback": False,
                        },
                        {
                            "provider": "fal_ai",
                            "model": "fal-ai/flux/schnell",
                            "status": "failed",
                            "error_code": "configuration_error",
                            "duration_ms": 510,
                            "fallback": True,
                        },
                    ],
                    "primary_provider": "stability_ai",
                    "fallback_provider": "fal_ai",
                    "fallback_provider_attempted": True,
                    "fallback_provider_used": False,
                }
            },
        },
        node_name="image_agent_node",
        status="degraded",
        node_started_at="2026-05-21T15:00:00.000+00:00",
        node_ended_at="2026-05-21T15:00:00.930+00:00",
        duration_ms=930,
    )

    assert metadata["provider_call_count_by_provider"] == {
        "stability_ai": 1,
        "fal_ai": 1,
    }
    assert metadata["provider_latency_by_provider_ms"] == {
        "stability_ai": 420,
        "fal_ai": 510,
    }
    assert metadata["primary_provider"] == "stability_ai"
    assert metadata["fallback_provider"] == "fal_ai"
    assert metadata["fallback_provider_attempted"] is True
    assert metadata["fallback_provider_used"] is False
    assert (
        metadata["image_provider_attempts"][0]["error_code"]
        == "authentication_failed"
    )
    assert metadata["image_provider_attempts"][1]["error_code"] == "configuration_error"
