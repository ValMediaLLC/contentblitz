from contentblitz.agents import quality_validator as quality_validator_module
from contentblitz.state import create_initial_state


def _base_state(**overrides):
    state = create_initial_state(
        requested_outputs=["blog"],
        content_drafts={
            "blog": {"body": "Draft body for validation testing.", "version": 2},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        },
        draft_status={"blog": "complete"},
        best_drafts={"blog": None, "linkedin": None},
        attempt_history={"blog": [], "linkedin": [], "image": []},
        errors=[],
    )
    state.update(overrides)
    return state


def test_passed_when_composite_at_or_above_threshold(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.75}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.75
    assert updates["quality_scores"]["blog"]["passed"] is True
    assert updates["quality_scores"]["blog"]["validation_status"] == "passed"
    assert updates["retry_requested"] is False
    assert updates["retry_target"] == ""


def test_retry_needed_when_composite_in_mid_range(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.50}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.5
    assert updates["quality_scores"]["blog"]["passed"] is False
    assert updates["quality_scores"]["blog"]["validation_status"] == "retry_needed"
    assert updates["retry_requested"] is True
    assert updates["retry_target"] == "blog"


def test_failed_when_composite_below_mid_range(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.49}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.49
    assert updates["quality_scores"]["blog"]["passed"] is False
    assert updates["quality_scores"]["blog"]["validation_status"] == "failed"
    assert updates["retry_requested"] is False
    assert updates["retry_target"] == ""


def test_unverified_fallback_when_scoring_errors(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        raise RuntimeError("validator unavailable")

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.6
    assert updates["quality_scores"]["blog"]["passed"] is False
    assert updates["quality_scores"]["blog"]["validation_status"] == "unverified"
    assert updates["retry_requested"] is False


def test_best_drafts_updates_only_when_score_improves(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.80}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )

    state = _base_state(
        best_drafts={
            "blog": {
                "version": 1,
                "body": "Prior best draft",
                "composite": 0.90,
                "validation_status": "passed",
            },
            "linkedin": None,
        }
    )
    updates = quality_validator_module.quality_validator_node(state)
    assert updates["best_drafts"]["blog"]["composite"] == 0.90
    assert updates["best_drafts"]["blog"]["body"] == "Prior best draft"

    state = _base_state(best_drafts={"blog": None, "linkedin": None})
    updates = quality_validator_module.quality_validator_node(state)
    assert updates["best_drafts"]["blog"]["composite"] == 0.80
    assert updates["best_drafts"]["blog"]["version"] == 2


def test_attempt_history_copies_current_draft_version(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.82}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    updates = quality_validator_module.quality_validator_node(_base_state())

    latest_attempt = updates["attempt_history"]["blog"][-1]
    assert latest_attempt["version"] == 2
    assert latest_attempt["composite"] == 0.82


def test_missing_draft_is_skipped_with_non_blocking_error(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        raise AssertionError(
            "validate_content should not be called when body is missing"
        )

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    state = _base_state(
        content_drafts={
            "blog": {"body": "   ", "version": 3},
            "linkedin": {"body": "", "version": 0},
            "research_report": {"body": ""},
        }
    )
    updates = quality_validator_module.quality_validator_node(state)

    assert "blog" not in updates["quality_scores"]
    assert updates["attempt_history"]["blog"] == []
    assert updates["retry_requested"] is False
    assert updates["retry_target"] == ""
    assert updates["errors"][-1] == {
        "agent": "quality_validator",
        "type": "missing_draft",
        "message": "No draft body available for blog.",
        "recoverable": True,
    }


def test_node_does_not_route_directly(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.80}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert "routing_decision" not in updates
    assert "next_node" not in updates


def test_incoming_retry_flag_does_not_force_retry_when_score_passes(
    monkeypatch,
) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.80}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    state = _base_state(retry_requested=True, retry_target="blog")
    updates = quality_validator_module.quality_validator_node(state)
    assert updates["quality_scores"]["blog"]["validation_status"] == "passed"
    assert updates["retry_requested"] is False
    assert updates["retry_target"] == ""


def test_citation_validation_adds_safe_warning_for_invalid_sources(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.8}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    state = _base_state(
        research_required=True,
        sources=[
            {
                "title": "Unsafe Source",
                "url": "javascript:alert(1)",
                "snippet": "Unsafe link in source model.",
            }
        ],
    )
    updates = quality_validator_module.quality_validator_node(state)

    citation_meta = updates["quality_scores"]["citation_validation"]
    assert citation_meta["status"] == "degraded"
    assert citation_meta["unsafe_url_count"] == 1
    assert any(
        "citation validation found" in msg.lower() for msg in updates["status_messages"]
    )


def test_citation_validation_does_not_add_warning_for_valid_sources(
    monkeypatch,
) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.8}

    monkeypatch.setattr(
        quality_validator_module, "validate_content", fake_validate_content
    )
    state = _base_state(
        research_required=True,
        status_messages=[],
        sources=[
            {
                "title": "Safe Source",
                "url": "https://example.com/safe-source",
                "snippet": "Credible citation text.",
            }
        ],
    )
    updates = quality_validator_module.quality_validator_node(state)

    citation_meta = updates["quality_scores"]["citation_validation"]
    assert citation_meta["status"] == "passed"
    assert "status_messages" not in updates
