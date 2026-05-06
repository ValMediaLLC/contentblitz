"""
Development smoke test utility.

Purpose:
- manually execute graph flows
- validate routing behavior
- inspect state transitions
- smoke test research_agent_node behavior

Not part of production runtime.
"""

from pathlib import Path
import sys
from pprint import pprint

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    query = input("Enter prompt: ").strip()

    state = create_initial_state(user_query=query)

    graph = build_langgraph()
    result = graph.invoke(state)

    print_section("ROUTING RESULT")

    routing_keys = [
        "intent",
        "requested_outputs",
        "research_required",
        "clarification_needed",
        "routing_decision",
        "workflow_status",
        "final_response",
    ]

    for key in routing_keys:
        print(f"{key}: {result.get(key)}")

    print_section("RESEARCH DATA")

    research_data = result.get("research_data") or {}
    print("research_data exists:", bool(research_data))
    print("summary:", research_data.get("synthesized_summary"))
    print("quality:", research_data.get("quality"))
    print("key_facts:")
    pprint(research_data.get("key_facts", []))
    print("keywords:")
    pprint(research_data.get("keywords", []))

    print_section("SOURCES")

    sources = result.get("sources") or []
    print("source_count:", len(sources))

    for index, source in enumerate(sources[:5], start=1):
        print(f"\nSource {index}")
        pprint(source)

    print_section("CACHE")

    pprint(result.get("cache_metadata"))

    print_section("COST CONTROLS")

    pprint(result.get("cost_controls"))

    print_section("ERRORS")

    errors = result.get("errors") or []
    if not errors:
        print("No errors.")
    else:
        pprint(errors)


if __name__ == "__main__":
    main()