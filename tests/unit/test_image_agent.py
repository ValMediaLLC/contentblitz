from contentblitz.agents import image_agent as image_agent_module
from contentblitz.state import create_initial_state


def _base_state(**overrides):
    state = create_initial_state(
        user_query="Create an image of a futuristic marketing dashboard",
        research_data={
            "synthesized_summary": (
                "Marketing teams rely on AI-assisted KPI forecasting."
            ),
            "keywords": ["ai", "dashboard", "forecasting"],
        },
        content_brief={
            "blog": {},
            "linkedin": {},
            "image": {
                "prompt_focus": "futuristic marketing command center",
                "visual_direction": "cinematic lighting",
                "style": "editorial",
            },
        },
    )
    state.update(overrides)
    return state


def test_cost_cap_skips_generation(monkeypatch) -> None:
    calls = {"image": 0}

    def fake_generate_image(prompt, style="default"):
        calls["image"] += 1
        return {"images": [{"url": "https://example.com/image.png"}]}

    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    state = _base_state(
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 2,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "image_generation_cap_per_session": 2,
        }
    )
    updates = image_agent_module.image_agent_node(state)
    assert calls["image"] == 0
    assert updates["tool_outputs"]["image_agent"]["status"] == "skipped"


def test_content_brief_image_is_preferred(monkeypatch) -> None:
    seen = {"prompt": ""}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": ""}

    def fake_generate_image(prompt, style="default"):
        seen["prompt"] = prompt
        return {"images": [{"url": "https://example.com/brief.png"}]}

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    image_agent_module.image_agent_node(_base_state())
    assert "futuristic marketing command center" in seen["prompt"].lower()


def test_missing_brief_derives_concept(monkeypatch) -> None:
    seen = {"prompt": ""}

    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": ""}

    def fake_generate_image(prompt, style="default"):
        seen["prompt"] = prompt
        return {"images": [{"url": "https://example.com/derived.png"}]}

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    state = _base_state(
        content_brief={"blog": {}, "linkedin": {}, "image": {}},
        user_query="Create an image about AI retail analytics",
        research_data={
            "synthesized_summary": "Retail teams use predictive models for inventory.",
            "keywords": ["retail", "inventory", "predictive"],
        },
    )
    image_agent_module.image_agent_node(state)
    assert "ai retail analytics" in seen["prompt"].lower()
    assert "retail teams use predictive models" in seen["prompt"].lower()


def test_successful_image_writes_image_outputs(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Enhanced futuristic dashboard scene with clean lines."}

    def fake_generate_image(prompt, style="default"):
        return {
            "provider_used": "dall-e-3",
            "images": [
                {
                    "url": "https://example.com/success.png",
                    "id": "img_1",
                    "width": 1024,
                    "height": 1024,
                }
            ],
        }

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    updates = image_agent_module.image_agent_node(_base_state())

    assert updates["image_prompts"]
    assert updates["image_outputs"]
    assert updates["image_outputs"][0]["status"] == "success"
    assert updates["image_outputs"][0]["url"] == "https://example.com/success.png"
    assert updates["image_outputs"][0]["renderable"] is True
    assert updates["draft_status"]["image"] == "complete"


def test_successful_local_path_image_writes_renderable_output(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Enhanced futuristic dashboard scene with clean lines."}

    def fake_generate_image(prompt, style="default"):
        return {
            "provider_used": "gpt-image-1",
            "images": [
                {
                    "local_path": "exports/images/fashion_001.png",
                    "id": "img_1",
                    "width": 1024,
                    "height": 1024,
                    "renderable": True,
                }
            ],
        }

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    updates = image_agent_module.image_agent_node(_base_state())

    assert updates["image_outputs"][0]["status"] == "success"
    assert updates["image_outputs"][0]["local_path"] == "exports/images/fashion_001.png"
    assert updates["image_outputs"][0]["renderable"] is True
    assert updates["draft_status"]["image"] == "complete"


def test_failed_image_writes_recoverable_error(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Enhanced prompt."}

    def fake_generate_image(prompt, style="default"):
        return {"provider_used": "dall-e-3", "images": [], "error": "provider timeout"}

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    updates = image_agent_module.image_agent_node(_base_state())

    assert updates["image_outputs"][-1]["status"] == "failed"
    assert updates["image_outputs"][-1]["recoverable"] is True
    assert updates["image_outputs"][-1]["error"]["code"] == "unknown_provider_error"
    assert "traceback" not in str(updates["image_outputs"][-1]["error"]).lower()
    assert updates["tool_outputs"]["image_agent"]["status"] == "failed"
    assert updates["draft_status"]["image"] == "failed"
    assert len(updates["image_prompts"]) >= 1
    assert len(updates["image_outputs"]) >= 1
    assert updates["cost_controls"]["image_generations_used_this_session"] == 0
    assert len(updates["errors"]) == 1
    assert updates["errors"][0] == {
        "agent": "image_agent",
        "type": "image_generation_failed",
        "message": (
            "Image generation encountered a recoverable issue. "
            "Text/research/export outputs remain available."
        ),
        "recoverable": True,
    }


def test_base64_is_never_stored(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Enhanced prompt with atmosphere."}

    def fake_generate_image(prompt, style="default"):
        return {
            "images": [
                {
                    "url": "https://example.com/asset.png",
                    "b64_json": "AAAABBBBCCCC",
                    "base64": "AAAABBBBCCCC",
                    "revised_prompt": "refined prompt",
                }
            ]
        }

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    updates = image_agent_module.image_agent_node(_base_state())
    payload = updates["image_outputs"][0]
    assert "b64_json" not in payload
    assert "base64" not in payload


def test_non_renderable_asset_id_is_degraded_not_success(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Enhanced prompt with atmosphere."}

    def fake_generate_image(prompt, style="default"):
        return {
            "provider_used": "gpt-image-1",
            "images": [
                {
                    "id": "img_asset_only_001",
                    "renderable": False,
                }
            ],
        }

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    updates = image_agent_module.image_agent_node(_base_state())

    payload = updates["image_outputs"][0]
    assert payload["status"] == "degraded"
    assert payload["renderable"] is False
    assert payload["id"] == "img_asset_only_001"
    assert updates["tool_outputs"]["image_agent"]["status"] == "degraded"
    assert updates["draft_status"]["image"] == "failed"


def test_image_counter_increments_only_on_success(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Enhanced prompt."}

    def fake_generate_image_success(prompt, style="default"):
        return {"images": [{"url": "https://example.com/a.png"}]}

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(
        image_agent_module, "generate_image", fake_generate_image_success
    )
    success_state = _base_state(
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 1,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
        }
    )
    success_updates = image_agent_module.image_agent_node(success_state)
    assert success_updates["cost_controls"]["image_generations_used_this_session"] == 2

    def fake_generate_image_failure(prompt, style="default"):
        return {"images": [], "error": "temp unavailable"}

    monkeypatch.setattr(
        image_agent_module, "generate_image", fake_generate_image_failure
    )
    failure_state = _base_state(
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 1,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
        }
    )
    failure_updates = image_agent_module.image_agent_node(failure_state)
    assert failure_updates["cost_controls"]["image_generations_used_this_session"] == 1


def test_no_assets_failure_writes_expected_error_message(monkeypatch) -> None:
    def fake_generate_text(prompt, agent_key, model="gpt-4o", metadata=None):
        return {"output": "Enhanced prompt."}

    def fake_generate_image(prompt, style="default"):
        return {"images": []}

    monkeypatch.setattr(image_agent_module, "generate_text", fake_generate_text)
    monkeypatch.setattr(image_agent_module, "generate_image", fake_generate_image)
    updates = image_agent_module.image_agent_node(_base_state())

    assert updates["errors"][0]["message"] == (
        "Image generation encountered a recoverable issue. "
        "Text/research/export outputs remain available."
    )
    assert updates["errors"][0]["recoverable"] is True
