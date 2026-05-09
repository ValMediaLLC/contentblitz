import sys
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from contentblitz.agents.research_agent import research_agent_node
from contentblitz.state import create_initial_state
from contentblitz.tools.cache import clear_cache


def build_state(query):
    state = create_initial_state(user_query=query)
    state["research_required"] = True
    state["cache_metadata"]["enabled"] = True
    return state


def main():
    query = input("Enter research query: ").strip() or "latest AI content marketing trends"
    clear_cache()

    print("\n==============================")
    print("RUN 1 - EXPECTED CACHE MISS")
    print("==============================")

    first = build_state(query)
    first_updates = research_agent_node(first)
    first.update(first_updates)

    first_sources = len(first.get("sources", []))
    first_cache_keys = first.get("cache_metadata", {}).get("keys", [])
    first_searches = first.get("cost_controls", {}).get("search_queries_used_this_session", 0)

    print("Sources:", first_sources)
    print("Cache keys:", first_cache_keys)
    print("Searches used:", first_searches)

    print("\n==============================")
    print("RUN 2 - EXPECTED CACHE HIT")
    print("==============================")

    second = build_state(query)

    def fail_if_search_called(*args, **kwargs):
        raise AssertionError("FAIL: search_web was called on Run 2. Cache was not used.")

    try:
        with patch(
            "contentblitz.agents.research_agent.search_web",
            side_effect=fail_if_search_called,
        ):
            second_updates = research_agent_node(second)
            second.update(second_updates)

        second_sources = len(second.get("sources", []))
        second_cache_keys = second.get("cache_metadata", {}).get("keys", [])
        second_searches = second.get("cost_controls", {}).get("search_queries_used_this_session", 0)

        print("Sources:", second_sources)
        print("Cache keys:", second_cache_keys)
        print("Searches used:", second_searches)

        print("\n==============================")
        print("CACHE VALIDATION")
        print("==============================")

        if second_sources > 0 and second_searches == 0:
            print("PASS: Run 2 used cache and did not call search_web.")
            print("FINAL RESULT: CACHE BEHAVIOR IS CORRECT")
        else:
            print("FAIL: Run 2 completed but cache indicators are ambiguous.")
            print("FINAL RESULT: CACHE VALIDATION FAILED")

    except AssertionError as exc:
        print(str(exc))
        print("FINAL RESULT: CACHE VALIDATION FAILED")


if __name__ == "__main__":
    main()
