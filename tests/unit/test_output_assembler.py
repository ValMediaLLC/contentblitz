from copy import deepcopy

from contentblitz.agents import output_assembler as output_assembler_module
from contentblitz.state import create_initial_state


def _base_state(**overrides):
    state = create_initial_state(
        user_query="AI content systems",
        requested_outputs=["blog"],
        content_drafts={
            "blog": {"body": "Current blog draft body.", "version": 1},
            "linkedin": {"body": "Current linkedin draft body.", "version": 1},
            "research_report": {"body": ""},
        },
        best_drafts={
            "blog": {"body": "Best blog draft body.", "composite": 0.90, "version": 2},
            "linkedin": {
                "body": "Best linkedin draft body.",
                "composite": 0.91,
                "version": 2,
            },
        },
        quality_scores={
            "blog": {"composite": 0.80, "validation_status": "passed", "passed": True},
            "linkedin": {
                "composite": 0.80,
                "validation_status": "passed",
                "passed": True,
            },
        },
        sources=[
            {
                "title": "Source A",
                "url": "https://example.com/a",
                "credibility_score": 0.6,
                "citation_available": True,
            }
        ],
        image_outputs=[],
        errors=[],
        export_metadata={
            "formats_requested": [],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        },
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
        },
    )
    state.update(overrides)
    return state


def test_research_only_report_generated_inline_when_report_missing() -> None:
    state = _base_state(
        requested_outputs=["research"],
        content_drafts={
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        research_data={
            "synthesized_summary": "AI research summary.",
            "key_facts": ["Fact 1", "Fact 2", "Fact 3"],
        },
    )
    original_sources = deepcopy(state["sources"])
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "success"
    assert updates["final_response"]
    assert "Research Report" in updates["final_response"]
    assert "AI research summary." in updates["final_response"]
    assert state["sources"] == original_sources
    assert "sources" not in updates


def test_blog_assembled() -> None:
    state = _base_state(requested_outputs=["blog"])
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "success"
    assert "## Blog Draft" in updates["final_response"]
    assert "Best blog draft body." in updates["final_response"]


def test_linkedin_assembled() -> None:
    state = _base_state(requested_outputs=["linkedin"])
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "success"
    assert "## LinkedIn Draft" in updates["final_response"]
    assert "Best linkedin draft body." in updates["final_response"]


def test_image_reference_included_when_image_assets_exist() -> None:
    state = _base_state(
        requested_outputs=["image"],
        image_outputs=[
            {
                "status": "success",
                "url": "https://example.com/image.png",
                "id": "img_123",
            }
        ],
    )
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "success"
    assert "## Image Assets" in updates["final_response"]
    assert "https://example.com/image.png" in updates["final_response"]


def test_image_failure_warning_included() -> None:
    state = _base_state(
        requested_outputs=["image"],
        image_outputs=[
            {
                "status": "failed",
                "error": "No image assets returned.",
            }
        ],
        errors=[
            {
                "agent": "image_agent",
                "type": "image_generation_failed",
                "recoverable": True,
            }
        ],
    )
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "partial_success"
    assert "recoverable issue" in updates["final_response"].lower()


def test_image_only_with_renderable_local_path_is_success() -> None:
    state = _base_state(
        requested_outputs=["image"],
        image_prompts=["Create a clean fashion campaign visual."],
        image_outputs=[
            {
                "status": "success",
                "provider": "gpt-image-1",
                "local_path": "exports/images/campaign.png",
                "renderable": True,
            }
        ],
        sources=[],
    )
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "success"
    assert "## Image Assets" in updates["final_response"]
    assert "exports/images/campaign.png" in updates["final_response"]


def test_source_dedupe_preserves_order_and_highest_credibility() -> None:
    state = _base_state(
        requested_outputs=["blog"],
        sources=[
            {
                "title": "Low Cred Duplicate URL",
                "url": "https://example.com/dup",
                "credibility_score": 0.3,
                "citation_available": True,
            },
            {
                "title": "High Cred Duplicate URL",
                "url": "https://example.com/dup",
                "credibility_score": 0.9,
                "citation_available": True,
            },
            {
                "title": "Title Duplicate",
                "url": None,
                "credibility_score": 0.2,
                "citation_available": False,
            },
            {
                "title": "Title Duplicate",
                "url": None,
                "credibility_score": 0.8,
                "citation_available": False,
            },
        ],
    )
    updates = output_assembler_module.output_assembler_node(state)

    response = updates["final_response"]
    assert "## Sources" in response
    assert "[1] High Cred Duplicate URL (https://example.com/dup)" in response
    assert response.count("Title Duplicate") == 1


def test_failed_or_unverified_quality_creates_partial_success() -> None:
    state = _base_state(
        requested_outputs=["blog"],
        quality_scores={
            "blog": {"composite": 0.40, "validation_status": "failed", "passed": False},
        },
    )
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "partial_success"
    assert "quality warnings" in updates["final_response"].lower()
    assert updates["final_response"].strip()


def test_no_usable_content_creates_failed() -> None:
    state = _base_state(
        requested_outputs=["blog", "linkedin"],
        content_drafts={
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        best_drafts={"blog": None, "linkedin": None},
        image_outputs=[],
        sources=[],
    )
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "failed"
    assert updates["final_response"].strip()


def test_final_response_non_empty_on_success_or_partial_success() -> None:
    success_state = _base_state(requested_outputs=["blog"])
    success_updates = output_assembler_module.output_assembler_node(success_state)
    assert success_updates["workflow_status"] == "success"
    assert success_updates["final_response"].strip()

    partial_state = _base_state(
        requested_outputs=["blog"],
        quality_scores={
            "blog": {
                "composite": 0.50,
                "validation_status": "unverified",
                "passed": False,
            },
        },
    )
    partial_updates = output_assembler_module.output_assembler_node(partial_state)
    assert partial_updates["workflow_status"] == "partial_success"
    assert partial_updates["final_response"].strip()


def test_export_requested_set_when_formats_exist() -> None:
    state = _base_state(
        export_requested=False,
        export_metadata={
            "formats_requested": ["markdown", "html"],
            "export_paths": {},
            "exported_at": None,
            "error_log": [],
        },
    )
    updates = output_assembler_module.output_assembler_node(state)
    assert updates["export_requested"] is True


def test_fallback_text_content_sets_partial_success_and_warning() -> None:
    state = _base_state(
        requested_outputs=["blog", "linkedin"],
        content_drafts={
            "blog": {
                "body": "## Fallback Blog Outline\nLimited blog draft.",
                "version": 1,
                "fallback_generated": True,
                "degraded_generation": True,
                "provider_failure_reason": "quota_exceeded",
            },
            "linkedin": {
                "body": "Fallback LinkedIn draft.",
                "version": 1,
                "fallback_generated": True,
                "degraded_generation": True,
            },
            "research_report": {"body": ""},
        },
        best_drafts={"blog": None, "linkedin": None},
        quality_scores={
            "blog": {
                "validation_status": "degraded",
                "passed": False,
                "composite": 0.6,
            },
            "linkedin": {
                "validation_status": "degraded",
                "passed": False,
                "composite": 0.6,
            },
        },
    )
    updates = output_assembler_module.output_assembler_node(state)

    assert updates["workflow_status"] == "partial_success"
    assert "fallback draft content is limited" in updates["final_response"].lower()
    assert updates["assembled_outputs"]["text_generation_degraded"] is True
    assert updates["assembled_outputs"]["fallback_content_used"] is True
    assert updates["assembled_outputs"]["real_generation_succeeded"] is False
    assert updates["assembled_outputs"]["provider_failure_reason"] == "quota_exceeded"
