"""
Development forced-retry scenario utility.

Purpose:
- manually validate retry_router_node behavior
- test retry_needed routing without relying on natural prompt output
- verify retry counts, retry feedback, and session retry caps

Not part of production runtime.
"""

from pathlib import Path
import sys
from pprint import pprint

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contentblitz.state import create_initial_state
from contentblitz.agents.retry_router import retry_router_node
from contentblitz.workflow.routing import route_from_retry_router
from contentblitz.config import RETRY_POLICY


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def run_scenario(name: str, state: dict) -> None:
    print_section(name)

    updated = retry_router_node(state)
    merged = {**state, **updated}

    route = route_from_retry_router(merged)

    print("route_from_retry_router:", route)
    print("retry_counts:")
    pprint(merged.get("retry_counts"))
    print("retry_feedback:")
    pprint(merged.get("retry_feedback"))
    print("cost_controls:")
    pprint(merged.get("cost_controls"))
    print("errors:")
    pprint(merged.get("errors"))


def scenario_blog_retry_needed() -> None:
    state = create_initial_state(
        user_query="create a blog article about AI workflow automation",
        requested_outputs=["blog"],
    )

    state["quality_scores"] = {
        "blog": {
            "composite": 0.62,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve clarity and add stronger structure."],
        }
    }

    run_scenario("SCENARIO 1 — Blog retry needed", state)


def scenario_linkedin_retry_needed() -> None:
    state = create_initial_state(
        user_query="write a linkedin post about AI marketing systems",
        requested_outputs=["linkedin"],
    )

    state["quality_scores"] = {
        "linkedin": {
            "composite": 0.64,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Tighten the hook and improve CTA clarity."],
        }
    }

    run_scenario("SCENARIO 2 — LinkedIn retry needed", state)


def scenario_both_retry_needed() -> None:
    state = create_initial_state(
        user_query="write a blog article and linkedin post about AI systems",
        requested_outputs=["blog", "linkedin"],
    )

    state["quality_scores"] = {
        "blog": {
            "composite": 0.61,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve structure and add clearer takeaways."],
        },
        "linkedin": {
            "composite": 0.63,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve hook and make CTA more specific."],
        },
    }

    run_scenario("SCENARIO 3 — Blog + LinkedIn retry needed", state)


def scenario_blog_max_retry_reached() -> None:
    state = create_initial_state(
        user_query="create a blog article about AI workflow automation",
        requested_outputs=["blog"],
    )

    state["retry_counts"]["blog_writer"] = RETRY_POLICY["blog_writer"]

    state["quality_scores"] = {
        "blog": {
            "composite": 0.62,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve clarity and add stronger structure."],
        }
    }

    run_scenario("SCENARIO 4 — Blog max retry reached", state)


def scenario_session_retry_cap_reached() -> None:
    state = create_initial_state(
        user_query="create a blog article about AI workflow automation",
        requested_outputs=["blog"],
    )

    state["cost_controls"]["total_retries_used_this_session"] = state[
        "cost_controls"
    ].get("max_total_retries_per_session", 3)

    state["quality_scores"] = {
        "blog": {
            "composite": 0.62,
            "passed": False,
            "validation_status": "retry_needed",
            "feedback": ["Improve clarity and add stronger structure."],
        }
    }

    run_scenario("SCENARIO 5 — Session retry cap reached", state)


def main() -> None:
    scenario_blog_retry_needed()
    scenario_linkedin_retry_needed()
    scenario_both_retry_needed()
    scenario_blog_max_retry_reached()
    scenario_session_retry_cap_reached()


if __name__ == "__main__":
    main()
