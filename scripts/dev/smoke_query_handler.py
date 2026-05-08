"""
Development smoke test utility.

Purpose:
- manually execute graph flows
- validate routing behavior
- inspect state transitions
- smoke test query_handler_node, research_agent_node, and content_strategist_node

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

    print_section("CONTENT BRIEF")

    content_brief = result.get("content_brief") or {}
    print("content_brief exists:", bool(content_brief))
    print("blog brief:")
    pprint(content_brief.get("blog", {}))
    print("linkedin brief:")
    pprint(content_brief.get("linkedin", {}))
    print("image brief:")
    pprint(content_brief.get("image", {}))

    print_section("CONTENT DRAFTS")

    content_drafts = result.get("content_drafts") or {}
    research_report = (content_drafts.get("research_report") or {})
    blog_draft = content_drafts.get("blog") or {}
    linkedin_draft = content_drafts.get("linkedin") or {}

    research_report_title = research_report.get("title")
    research_report_body = (research_report.get("body") or "").strip()
    research_report_populated = bool(research_report_title or research_report_body)

    print("research_report populated:", research_report_populated)
    print("research_report title:", research_report_title)
    print("research_report body length:", len(research_report_body))

    print("\nblog draft:")
    pprint(blog_draft)

    print("\nlinkedin draft:")
    pprint(linkedin_draft)
    print("\ndraft_status:")
    pprint(result.get("draft_status") or {})

    print_section("IMAGE OUTPUTS")

    image_prompts = result.get("image_prompts") or []
    image_outputs = result.get("image_outputs") or []

    print("image_prompt_count:", len(image_prompts))
    pprint(image_prompts)

    print("image_output_count:", len(image_outputs))
    pprint(image_outputs)

    print_section("QUALITY SCORES")
    pprint(result.get("quality_scores"))

    print_section("BEST DRAFTS")
    pprint(result.get("best_drafts"))

    print_section("ATTEMPT HISTORY")
    pprint(result.get("attempt_history"))

    print_section("FINAL OUTPUT")

    print("workflow_status:", result.get("workflow_status"))
    print("final_response:")
    print(result.get("final_response"))

    print("assembled_outputs:")
    pprint(result.get("assembled_outputs"))    

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
