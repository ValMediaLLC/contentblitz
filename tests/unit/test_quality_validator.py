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

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.75
    assert updates["quality_scores"]["blog"]["passed"] is True
    assert updates["quality_scores"]["blog"]["validation_status"] == "passed"
    assert updates["retry_requested"] is False
    assert updates["retry_target"] == ""


def test_retry_needed_when_composite_in_mid_range(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.50}

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.5
    assert updates["quality_scores"]["blog"]["passed"] is False
    assert updates["quality_scores"]["blog"]["validation_status"] == "retry_needed"
    assert updates["retry_requested"] is True
    assert updates["retry_target"] == "blog"


def test_failed_when_composite_below_mid_range(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.49}

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.49
    assert updates["quality_scores"]["blog"]["passed"] is False
    assert updates["quality_scores"]["blog"]["validation_status"] == "failed"
    assert updates["retry_requested"] is False
    assert updates["retry_target"] == ""


def test_unverified_fallback_when_scoring_errors(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        raise RuntimeError("validator unavailable")

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert updates["quality_scores"]["blog"]["composite"] == 0.6
    assert updates["quality_scores"]["blog"]["passed"] is False
    assert updates["quality_scores"]["blog"]["validation_status"] == "unverified"
    assert updates["retry_requested"] is False


def test_best_drafts_updates_only_when_score_improves(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        return {"composite": 0.80}

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)

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

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)
    updates = quality_validator_module.quality_validator_node(_base_state())

    latest_attempt = updates["attempt_history"]["blog"][-1]
    assert latest_attempt["version"] == 2
    assert latest_attempt["composite"] == 0.82


def test_missing_draft_is_skipped_with_non_blocking_error(monkeypatch) -> None:
    def fake_validate_content(content_type, draft_body, context=None):
        raise AssertionError("validate_content should not be called when body is missing")

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)
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

    monkeypatch.setattr(quality_validator_module, "validate_content", fake_validate_content)
    updates = quality_validator_module.quality_validator_node(_base_state())

    assert "routing_decision" not in updates
    assert "next_node" not in updates
