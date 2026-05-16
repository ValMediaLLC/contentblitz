import contentblitz

from contentblitz.config import (
    RETRY_POLICY,
    build_cache_metadata_defaults,
    build_cost_controls_defaults,
    live_provider_calls_enabled,
    validate_retry_policy_keys,
)
from contentblitz.state import create_initial_state


def test_package_imports_successfully() -> None:
    assert contentblitz is not None


def test_retry_policy_keys_exactly_match_retry_counts_keys() -> None:
    state = create_initial_state()
    assert validate_retry_policy_keys(state["retry_counts"])
    assert set(RETRY_POLICY.keys()) == set(state["retry_counts"].keys())


def test_default_builders_return_independent_objects() -> None:
    cache_a = build_cache_metadata_defaults()
    cache_b = build_cache_metadata_defaults()
    cache_a["keys"].append("k1")
    assert cache_b["keys"] == []

    cost_a = build_cost_controls_defaults()
    cost_b = build_cost_controls_defaults()
    cost_a["tokens_used_this_session"] = 99
    assert cost_b["tokens_used_this_session"] == 0


def test_runtime_flag_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", raising=False)

    assert live_provider_calls_enabled() is True


def test_runtime_flag_false_values_disable_features(monkeypatch) -> None:
    monkeypatch.setenv("CONTENTBLITZ_ENABLE_LIVE_CALLS", "false")

    assert live_provider_calls_enabled() is False
