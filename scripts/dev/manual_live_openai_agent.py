import sys
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

if os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") != "1":
    print(
        "Live API execution is disabled. "
        "Set CONTENTBLITZ_RUN_LIVE_TESTS=1 in .env to enable."
    )
    sys.exit(0)

from contentblitz.state import create_initial_state
from contentblitz.agents.query_handler import query_handler_node
from contentblitz.agents.content_strategist import content_strategist_node
from contentblitz.agents.blog_writer import blog_writer_node
from contentblitz.agents.linkedin_writer import linkedin_writer_node


def main():
    print("\nContentBlitz Live OpenAI Manual Agent Test")
    print("-----------------------------------------")

    user_prompt = input("\nEnter your prompt: ").strip()

    if not user_prompt:
        print("No prompt entered.")
        return

    state = create_initial_state(user_query=user_prompt)

    print("\nRunning query_handler_node...")
    updates = query_handler_node(state)
    state.update(updates)
    print("Intent:", state.get("intent"))
    print("Requested outputs:", state.get("requested_outputs"))
    print("Research required:", state.get("research_required"))
    print("Clarification needed:", state.get("clarification_needed"))

    if state.get("clarification_needed"):
        print("\nClarification:")
        print(state.get("clarification_message"))
        return

    # Manual shortcut for now while Phase 2 integrations are still in progress.
    # This avoids requiring SERP/research before testing OpenAI writing agents.
    if not state.get("research_data"):
        state["research_data"] = {
            "summary": "Manual live OpenAI test. No external research was performed.",
            "key_points": [
                "Use this run to validate real OpenAI agent behavior.",
                "Research integration will be tested separately once SERP is implemented.",
            ],
        }

    print("\nRunning content_strategist_node...")
    updates = content_strategist_node(state)
    state.update(updates)

    print("\nContent brief:")
    print(state.get("content_brief"))

    if "blog" in state.get("requested_outputs", []):
        print("\nRunning blog_writer_node...")
        updates = blog_writer_node(state)
        state.update(updates)

        print("\nBLOG OUTPUT")
        print("-----------")
        print(state["content_drafts"]["blog"]["body"])

    if "linkedin" in state.get("requested_outputs", []):
        print("\nRunning linkedin_writer_node...")
        updates = linkedin_writer_node(state)
        state.update(updates)

        print("\nLINKEDIN OUTPUT")
        print("---------------")
        print(state["content_drafts"]["linkedin"]["body"])

    print("\nCost controls:")
    print(state.get("cost_controls"))


if __name__ == "__main__":
    main()