from __future__ import annotations

from copy import deepcopy

from contentblitz.agents.image_agent import image_agent_node
from contentblitz.agents.output_assembler import output_assembler_node
from contentblitz.state import create_initial_state


def test_budget_exceeded_notice_appears_in_assembled_output() -> None:
    state = create_initial_state(
        requested_outputs=["blog"],
        content_drafts={
            "blog": {"body": "Draft body for budget notice validation.", "version": 1},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        best_drafts={"blog": None, "linkedin": None},
        cost_controls={
            "tokens_used_this_session": 1200,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": True,
            "token_budget_per_session": 1000,
        },
    )

    updates = output_assembler_node(state)
    assert updates["workflow_status"] == "partial_success"
    assert (
        "Notice: Session budget was exceeded during generation."
        in updates["final_response"]
    )


def test_partial_success_possible_when_image_modality_is_blocked_by_cap() -> None:
    state = create_initial_state(
        requested_outputs=["blog", "image"],
        content_drafts={
            "blog": {"body": "Blog draft remains available.", "version": 1},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        best_drafts={"blog": None, "linkedin": None},
        quality_scores={
            "blog": {"composite": 0.8, "validation_status": "passed", "passed": True}
        },
        cost_controls={
            "tokens_used_this_session": 0,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 2,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "image_generation_cap_per_session": 2,
        },
    )

    image_updates = image_agent_node(state)
    merged = deepcopy(state)
    merged.update(image_updates)

    assembled = output_assembler_node(merged)
    assert assembled["workflow_status"] == "partial_success"
    assert "## Blog Draft" in assembled["final_response"]
    assert "recoverable failure" in assembled["final_response"].lower()
