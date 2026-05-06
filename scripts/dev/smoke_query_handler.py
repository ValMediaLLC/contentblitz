"""
Development smoke test utility.

Purpose:
- manually execute graph flows
- validate routing behavior
- inspect state transitions

Not part of production runtime.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contentblitz.state import create_initial_state
from contentblitz.workflow.graph import build_langgraph


def main():
    query = input("Enter prompt: ").strip()

    state = create_initial_state(user_query=query)

    graph = build_langgraph()

    result = graph.invoke(state)

    print("\n--- RESULT ---")

    keys = [
        "intent",
        "requested_outputs",
        "research_required",
        "clarification_needed",
        "routing_decision",
        "workflow_status",
        "final_response",
    ]

    for key in keys:
        print(f"{key}: {result.get(key)}")


if __name__ == "__main__":
    main()