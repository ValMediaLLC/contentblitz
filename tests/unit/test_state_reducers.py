from __future__ import annotations

from copy import deepcopy

import pytest

from contentblitz.state import (
    create_initial_state,
    merge_content_drafts as state_merge_content_drafts,
    merge_draft_status as state_merge_draft_status,
)
from contentblitz.workflow.graph import (
    merge_content_drafts,
    merge_cost_controls,
    merge_draft_status,
    merge_error_entries,
    merge_export_error_log,
    merge_export_metadata,
    merge_image_outputs,
    merge_nested_dict_skip_none,
    merge_progress_events,
    merge_retry_counts,
    merge_source_entries,
    merge_ui_node_statuses,
    merge_unique_text_list,
)


def _reduce_parallel_updates(base_state: dict, *branch_updates: dict) -> dict:
    """
    Simulate a deterministic fan-in reducer pass for reducer-owned fields.

    This mirrors how reducer-owned keys reconcile after fan-out branches return.
    """
    merged = deepcopy(base_state)
    for update in branch_updates:
        for key, value in update.items():
            if key == "content_drafts":
                merged[key] = merge_content_drafts(merged.get(key, {}), value)
                continue
            if key == "draft_status":
                merged[key] = merge_draft_status(merged.get(key, {}), value)
                continue
            if key == "cost_controls":
                merged[key] = merge_cost_controls(merged.get(key, {}), value)
                continue
            if key == "sources":
                merged[key] = merge_source_entries(merged.get(key, []), value)
                continue
            if key == "image_prompts":
                merged[key] = merge_unique_text_list(merged.get(key, []), value)
                continue
            if key == "image_outputs":
                merged[key] = merge_image_outputs(merged.get(key, []), value)
                continue
            if key == "quality_scores":
                merged[key] = merge_nested_dict_skip_none(merged.get(key, {}), value)
                continue
            if key == "retry_counts":
                merged[key] = merge_retry_counts(merged.get(key, {}), value)
                continue
            if key == "progress_events":
                merged[key] = merge_progress_events(merged.get(key, []), value)
                continue
            if key == "ui_node_statuses":
                merged[key] = merge_ui_node_statuses(merged.get(key, {}), value)
                continue
            if key == "warnings":
                merged[key] = merge_unique_text_list(merged.get(key, []), value)
                continue
            if key == "status_messages":
                merged[key] = merge_unique_text_list(merged.get(key, []), value)
                continue
            if key == "errors":
                merged[key] = merge_error_entries(merged.get(key, []), value)
                continue
            if key == "export_metadata":
                merged[key] = merge_export_metadata(merged.get(key, {}), value)
                continue
            merged[key] = value
    return merged


@pytest.mark.parametrize(
    "reducer",
    [merge_content_drafts, state_merge_content_drafts],
)
def test_content_draft_reducer_merges_nested_fields_without_wiping(reducer) -> None:
    left = {
        "blog": {"body": "Left body", "version": 1, "word_count": 120},
        "linkedin": {"body": "Left LI", "version": 1},
    }
    right = {
        "blog": {"version": 2, "word_count": None},
        "research_report": {"body": "Research summary"},
    }

    merged = reducer(left, right)

    assert merged["blog"]["body"] == "Left body"
    assert merged["blog"]["version"] == 2
    # None updates should not wipe existing nested values.
    assert merged["blog"]["word_count"] == 120
    assert merged["linkedin"]["body"] == "Left LI"
    assert merged["research_report"]["body"] == "Research summary"


@pytest.mark.parametrize(
    "reducer",
    [merge_content_drafts, state_merge_content_drafts],
)
def test_content_draft_reducer_ignores_none_branch_update(reducer) -> None:
    left = {"blog": {"body": "Keep me", "version": 4}}
    right = {"blog": None}

    merged = reducer(left, right)
    assert merged["blog"]["body"] == "Keep me"
    assert merged["blog"]["version"] == 4


@pytest.mark.parametrize(
    "reducer",
    [merge_draft_status, state_merge_draft_status],
)
def test_draft_status_reducer_keeps_existing_when_update_is_none(reducer) -> None:
    left = {"blog": "complete", "linkedin": "pending"}
    right = {"blog": None, "linkedin": "complete"}

    merged = reducer(left, right)
    assert merged == {"blog": "complete", "linkedin": "complete"}


def test_cost_controls_budget_exceeded_is_sticky_true() -> None:
    left = {
        "tokens_used_this_session": 200,
        "budget_exceeded": True,
    }
    right = {
        "tokens_used_this_session": 50,
        "budget_exceeded": False,
    }

    merged = merge_cost_controls(left, right)

    assert merged["tokens_used_this_session"] == 200
    assert merged["budget_exceeded"] is True


def test_cost_controls_counter_merge_is_order_independent_for_known_fields() -> None:
    left = {
        "tokens_used_this_session": 120,
        "search_queries_used_this_session": 2,
        "image_generations_used_this_session": 1,
        "total_retries_used_this_session": 1,
        "token_budget_per_session": 1000,
        "search_query_cap_per_session": 5,
        "image_generation_cap_per_session": 3,
        "max_total_retries_per_session": 4,
        "budget_exceeded": False,
    }
    right = {
        "tokens_used_this_session": 160,
        "search_queries_used_this_session": 1,
        "image_generations_used_this_session": 2,
        "total_retries_used_this_session": 2,
        "token_budget_per_session": 900,
        "search_query_cap_per_session": 4,
        "image_generation_cap_per_session": 2,
        "max_total_retries_per_session": 3,
        "budget_exceeded": True,
    }

    forward = merge_cost_controls(left, right)
    reverse = merge_cost_controls(right, left)

    assert forward == reverse
    assert forward["tokens_used_this_session"] == 160
    assert forward["search_queries_used_this_session"] == 2
    assert forward["image_generations_used_this_session"] == 2
    assert forward["total_retries_used_this_session"] == 2
    assert forward["token_budget_per_session"] == 900
    assert forward["search_query_cap_per_session"] == 4
    assert forward["image_generation_cap_per_session"] == 2
    assert forward["max_total_retries_per_session"] == 3
    assert forward["budget_exceeded"] is True


def test_cost_controls_identical_parallel_updates_do_not_double_count() -> None:
    base = {
        "tokens_used_this_session": 340,
        "search_queries_used_this_session": 4,
        "image_generations_used_this_session": 2,
        "total_retries_used_this_session": 1,
        "budget_exceeded": False,
    }

    merged_once = merge_cost_controls(base, base)
    merged_twice = merge_cost_controls(merged_once, base)

    assert merged_once["tokens_used_this_session"] == 340
    assert merged_once["search_queries_used_this_session"] == 4
    assert merged_once["image_generations_used_this_session"] == 2
    assert merged_once["total_retries_used_this_session"] == 1
    assert merged_twice == merged_once


def test_parallel_blog_and_linkedin_updates_reconcile_deterministically() -> None:
    base = create_initial_state(
        content_drafts={
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        draft_status={},
        cost_controls={
            "tokens_used_this_session": 20,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        },
    )

    blog_branch = {
        "content_drafts": {"blog": {"body": "Blog draft", "version": 1}},
        "draft_status": {"blog": "complete"},
        "cost_controls": {
            "tokens_used_this_session": 55,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        },
    }
    linkedin_branch = {
        "content_drafts": {"linkedin": {"body": "LinkedIn draft", "version": 1}},
        "draft_status": {"linkedin": "complete"},
        "cost_controls": {
            "tokens_used_this_session": 70,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        },
    }

    first = _reduce_parallel_updates(base, blog_branch, linkedin_branch)
    second = _reduce_parallel_updates(base, linkedin_branch, blog_branch)

    assert first == second
    assert first["content_drafts"]["blog"]["body"] == "Blog draft"
    assert first["content_drafts"]["linkedin"]["body"] == "LinkedIn draft"
    assert first["draft_status"] == {"blog": "complete", "linkedin": "complete"}
    assert first["cost_controls"]["tokens_used_this_session"] == 70


def test_parallel_blog_and_recoverable_image_failure_keep_text_outputs() -> None:
    base = create_initial_state(
        content_drafts={
            "blog": {"body": "", "version": 0},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        draft_status={},
        errors=[],
        cost_controls={
            "tokens_used_this_session": 30,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        },
    )

    blog_branch = {
        "content_drafts": {"blog": {"body": "Blog survives", "version": 1}},
        "draft_status": {"blog": "complete"},
        "cost_controls": {
            "tokens_used_this_session": 65,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        },
    }
    image_branch = {
        "draft_status": {"image": "failed"},
        "errors": [
            {
                "agent": "image_agent",
                "type": "image_generation_failed",
                "message": "Image generation encountered a recoverable issue.",
                "recoverable": True,
            }
        ],
        "cost_controls": {
            "tokens_used_this_session": 30,
            "search_queries_used_this_session": 0,
            "image_generations_used_this_session": 0,
            "total_retries_used_this_session": 0,
            "budget_exceeded": False,
            "token_budget_per_session": 1000,
        },
    }

    merged = _reduce_parallel_updates(base, blog_branch, image_branch)

    assert merged["content_drafts"]["blog"]["body"] == "Blog survives"
    assert merged["draft_status"]["blog"] == "complete"
    # Recoverable image branch details should not wipe successful text output.
    assert merged["draft_status"]["image"] == "failed"
    assert merged["errors"][0]["recoverable"] is True


def test_unique_text_reducer_dedupes_and_ignores_null_placeholders() -> None:
    merged = merge_unique_text_list(
        ["Research degraded", "None", "", "Retry used"],
        ["Retry used", "null", "Image fallback used"],
    )
    assert merged == ["Research degraded", "Retry used", "Image fallback used"]


def test_unique_text_reducer_ignores_non_string_entries() -> None:
    merged = merge_unique_text_list(
        ["Warning A", 123, {"message": "not allowed"}, False, None, "Warning B"],
        ["Warning B", 3.14, ["nested"], "Warning C"],
    )
    assert merged == ["Warning A", "Warning B", "Warning C"]


def test_error_reducer_dedupes_same_error_and_preserves_unique_entries() -> None:
    left = [
        {
            "agent": "image_agent",
            "type": "image_generation_failed",
            "message": "Image generation encountered a recoverable issue.",
            "recoverable": True,
        }
    ]
    right = [
        {
            "agent": "image_agent",
            "type": "image_generation_failed",
            "message": "Image generation encountered a recoverable issue.",
            "recoverable": True,
        },
        {
            "agent": "blog_writer",
            "type": "budget_exceeded",
            "message": "Blog generation used fallback due to token budget.",
            "recoverable": True,
        },
    ]
    merged = merge_error_entries(left, right)

    assert len(merged) == 2
    assert merged[0]["agent"] == "image_agent"
    assert merged[1]["agent"] == "blog_writer"


def test_export_error_log_reducer_keeps_structured_entries() -> None:
    left = [{"format": "markdown", "code": "markdown_validation_error", "message": "Missing title"}]
    right = [
        {"format": "markdown", "code": "markdown_validation_error", "message": "Missing title"},
        {"format": "html", "code": "html_export_failed", "message": "HTML export failed safely."},
    ]

    merged = merge_export_error_log(left, right)
    assert len(merged) == 2
    assert merged[0]["format"] == "markdown"
    assert merged[1]["format"] == "html"


def test_export_metadata_reducer_preserves_completed_format_results() -> None:
    left = {
        "formats_requested": ["markdown"],
        "export_paths": {"markdown": "exports/content_a.md"},
        "export_status": {"markdown": "completed"},
        "status_messages": ["Markdown export succeeded."],
        "error_log": [],
    }
    right = {
        "formats_requested": ["html"],
        "export_paths": {"html": "exports/content_a.html"},
        "export_status": {"html": "completed", "markdown": "failed"},
        "status_messages": ["HTML export succeeded.", "Markdown export succeeded."],
        "error_log": [{"format": "html", "code": "html_warning", "message": "Minor warning"}],
    }

    merged = merge_export_metadata(left, right)

    assert merged["formats_requested"] == ["markdown", "html"]
    assert merged["export_paths"]["markdown"] == "exports/content_a.md"
    assert merged["export_paths"]["html"] == "exports/content_a.html"
    # Completed must not be downgraded by conflicting concurrent branch updates.
    assert merged["export_status"]["markdown"] == "completed"
    assert merged["export_status"]["html"] == "completed"
    assert merged["status_messages"] == [
        "Markdown export succeeded.",
        "HTML export succeeded.",
    ]
    assert merged["error_log"][0]["format"] == "html"


def test_sources_reducer_dedupes_by_url_then_title_source_and_preserves_order() -> None:
    left = [
        {"title": "A", "url": "https://example.com/a", "source": "serp", "snippet": "left"},
        {"title": "B", "source": "perplexity", "snippet": "fallback"},
        "not-a-dict",
    ]
    right = [
        {"title": "A newer", "url": "https://example.com/a", "source": "serp", "snippet": "right"},
        {"title": "B", "source": "perplexity", "snippet": "dup by title+source"},
        {"title": "C", "url": "https://example.com/c", "source": "serp"},
        {"title": "", "url": "", "source": "", "snippet": ""},
    ]

    merged = merge_source_entries(left, right)

    assert len(merged) == 3
    assert merged[0]["url"] == "https://example.com/a"
    assert merged[1]["title"] == "B"
    assert merged[2]["url"] == "https://example.com/c"
    # Preserve left-first stable ordering and first valid occurrence.
    assert merged[0]["snippet"] == "left"


def test_image_prompts_reducer_dedupes_and_ignores_empty_or_null_tokens() -> None:
    merged = merge_unique_text_list(
        ["Prompt A", "Prompt A", "  ", "None", "null", "Prompt B"],
        ["Prompt B", "Prompt C", "", "None"],
    )
    assert merged == ["Prompt A", "Prompt B", "Prompt C"]


def test_image_outputs_reducer_preserves_success_and_recoverable_failure() -> None:
    left = [
        {"id": "img-1", "status": "success", "provider": "dall-e-3", "url": "https://img.example/1.png"},
        {"status": "failed", "provider": "dall-e-3", "prompt": "concept one", "error": "recoverable"},
    ]
    right = [
        {"id": "img-1", "status": "success", "provider": "dall-e-2", "url": "https://img.example/1.png"},
        {"status": "failed", "provider": "dall-e-3", "prompt": "concept one", "error": "recoverable"},
        {"status": "success", "provider": "dall-e-2", "url": "data:image/png;base64,ABC"},
        {"status": "success", "provider": "dall-e-2", "url": "https://img.example/2.png"},
    ]

    merged = merge_image_outputs(left, right)

    assert len(merged) == 3
    assert merged[0]["id"] == "img-1"
    assert merged[1]["status"] == "failed"
    assert merged[2]["url"] == "https://img.example/2.png"
    assert all("base64" not in str(item).lower() for item in merged)
    assert all(not str(item.get("url", "")).lower().startswith("data:image/") for item in merged)


def test_quality_scores_merge_keeps_unrelated_scores_and_skips_none() -> None:
    left = {
        "blog": {"composite": 0.8, "validation_status": "passed"},
        "citation_validation": {"status": "degraded", "invalid_count": 1},
    }
    right = {
        "linkedin": {"composite": 0.72, "validation_status": "passed"},
        "citation_validation": {"invalid_count": None, "duplicate_count": 2},
        "blog": None,
    }

    merged = merge_nested_dict_skip_none(left, right)

    assert merged["blog"]["composite"] == 0.8
    assert merged["linkedin"]["composite"] == 0.72
    assert merged["citation_validation"]["status"] == "degraded"
    assert merged["citation_validation"]["invalid_count"] == 1
    assert merged["citation_validation"]["duplicate_count"] == 2


def test_retry_counts_reducer_uses_per_key_max_and_ignores_invalid_values() -> None:
    left = {
        "blog_writer": 1,
        "linkedin_writer": 0,
        "image_agent": 2,
    }
    right = {
        "blog_writer": 3,
        "linkedin_writer": True,
        "image_agent": -4,
        "quality_validator": "oops",
    }

    merged = merge_retry_counts(left, right)

    assert merged["blog_writer"] == 3
    assert merged["linkedin_writer"] == 0
    assert merged["image_agent"] == 2
    assert "quality_validator" not in merged


def test_progress_events_reducer_is_stable_and_dedupes_exact_duplicates() -> None:
    left = [
        {
            "timestamp": "2026-05-13T10:00:00+00:00",
            "node_name": "query_handler_node",
            "status": "completed",
            "message": "done",
        },
        {"invalid": "entry"},
    ]
    right = [
        {
            "timestamp": "2026-05-13T10:00:00+00:00",
            "node_name": "query_handler_node",
            "status": "completed",
            "message": "done",
        },
        {
            "timestamp": "2026-05-13T10:00:01+00:00",
            "node_name": "blog_writer_node",
            "status": "running",
            "message": "blog running",
            "safe_metadata": {"attempt": 1, "raw": {"nested": "ignored"}},
        },
    ]

    merged = merge_progress_events(left, right)

    assert len(merged) == 2
    assert merged[0]["node_name"] == "query_handler_node"
    assert merged[1]["node_name"] == "blog_writer_node"
    assert merged[1]["safe_metadata"]["attempt"] == 1
    assert "raw" not in merged[1]["safe_metadata"]


def test_ui_node_statuses_precedence_prevents_pending_overwrite() -> None:
    left = {
        "blog_writer_node": "completed",
        "image_agent_node": "degraded",
        "quality_validator_node": "failed",
    }
    right = {
        "blog_writer_node": "pending",
        "image_agent_node": "running",
        "quality_validator_node": "completed",
        "linkedin_writer_node": "running",
    }

    merged = merge_ui_node_statuses(left, right)

    assert merged["blog_writer_node"] == "completed"
    assert merged["image_agent_node"] == "degraded"
    assert merged["quality_validator_node"] == "failed"
    assert merged["linkedin_writer_node"] == "running"
