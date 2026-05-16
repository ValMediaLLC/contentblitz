#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_PATHS = [
    "contentblitz",
    "contentblitz/__init__.py",
    "contentblitz/config.py",
    "contentblitz/state.py",
    "contentblitz/agents",
    "contentblitz/tools",
    "contentblitz/workflow",
    "contentblitz/core",
    "tests/unit",
    "tests/integration",
    "docs/ContentBlitz_Execution_Spec.md",
    "requirements.txt",
    ".gitignore",
    "README.md",
]

REQUIRED_REQUIREMENTS = {
    "langgraph",
    "openai",
    "pytest",
    "pytest-cov",
    "python-dotenv",
}

REQUIRED_GITIGNORE = {
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".env",
    ".pytest_cache/",
    "coverage*",
}

RETRY_KEYS = {
    "query_handler",
    "research_agent",
    "content_strategist",
    "blog_writer",
    "linkedin_writer",
    "image_agent",
    "quality_validator",
    "output_assembler",
    "export",
}

STATE_TOP_LEVEL_FIELDS = {
    "session_id",
    "user_id",
    "user_query",
    "intent",
    "routing_decision",
    "requested_outputs",
    "conversation_history",
    "research_required",
    "clarification_needed",
    "clarification_message",
    "research_data",
    "sources",
    "content_brief",
    "content_drafts",
    "draft_status",
    "best_drafts",
    "attempt_history",
    "retry_feedback",
    "retry_counts",
    "quality_scores",
    "image_prompts",
    "image_outputs",
    "tool_outputs",
    "errors",
    "final_response",
    "workflow_status",
    "export_requested",
    "export_metadata",
    "cache_metadata",
    "cost_controls",
}

AUTHORITATIVE_NODES = {
    "query_handler_node",
    "clarification_node",
    "research_agent_node",
    "content_strategist_node",
    "blog_writer_node",
    "linkedin_writer_node",
    "image_agent_node",
    "quality_validator_node",
    "retry_router_node",
    "output_assembler_node",
    "export_node",
    "error_handler_node",
}

EXPECTED_AGENT_MODULES = {
    "query_handler.py": "query_handler_node",
    "clarification.py": "clarification_node",
    "research_agent.py": "research_agent_node",
    "content_strategist.py": "content_strategist_node",
    "blog_writer.py": "blog_writer_node",
    "linkedin_writer.py": "linkedin_writer_node",
    "image_agent.py": "image_agent_node",
    "quality_validator.py": "quality_validator_node",
    "retry_router.py": "retry_router_node",
    "output_assembler.py": "output_assembler_node",
    "export.py": "export_node",
    "error_handler.py": "error_handler_node",
}


def fail(message: str) -> None:
    print(f"❌ {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"✅ {message}")


def warn(message: str) -> None:
    print(f"⚠️  {message}")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require_paths() -> None:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    if missing:
        fail("Missing required paths:\n" + "\n".join(f"  - {p}" for p in missing))
    ok("Required project structure exists")


def validate_requirements() -> None:
    content = read("requirements.txt").lower()
    missing = [dep for dep in REQUIRED_REQUIREMENTS if dep not in content]
    if missing:
        fail(
            "Missing required dependencies in requirements.txt:\n"
            + "\n".join(f"  - {d}" for d in missing)
        )
    ok("requirements.txt contains required dependencies")


def validate_gitignore() -> None:
    content = set(
        line.strip() for line in read(".gitignore").splitlines() if line.strip()
    )
    missing = REQUIRED_GITIGNORE - content
    if missing:
        fail(
            "Missing required .gitignore entries:\n"
            + "\n".join(f"  - {d}" for d in missing)
        )
    ok(".gitignore contains required entries")


def validate_no_unittest() -> None:
    test_files = list((ROOT / "tests").rglob("test_*.py"))
    offenders = []
    for file in test_files:
        text = file.read_text(encoding="utf-8")
        if "import unittest" in text or "unittest.TestCase" in text:
            offenders.append(str(file.relative_to(ROOT)))
    if offenders:
        fail(
            "unittest usage found. Convert these to pytest:\n"
            + "\n".join(f"  - {f}" for f in offenders)
        )
    ok("No unittest usage detected")


def load_module(path: str, name: str):
    full_path = ROOT / path
    spec = importlib.util.spec_from_file_location(name, full_path)
    if spec is None or spec.loader is None:
        fail(f"Could not load module: {path}")

    module = importlib.util.module_from_spec(spec)

    # Required for dataclasses / typing introspection
    sys.modules[name] = module

    spec.loader.exec_module(module)
    return module


def validate_config() -> None:
    module = load_module("contentblitz/config.py", "contentblitz_config")

    if not hasattr(module, "RETRY_POLICY"):
        fail("contentblitz/config.py must define RETRY_POLICY")

    retry_policy = module.RETRY_POLICY
    if set(retry_policy.keys()) != RETRY_KEYS:
        fail(
            "RETRY_POLICY keys do not match spec.\n"
            f"Expected: {sorted(RETRY_KEYS)}\n"
            f"Found:    {sorted(retry_policy.keys())}"
        )

    if not hasattr(module, "COST_CONTROLS_DEFAULTS"):
        fail("contentblitz/config.py must define COST_CONTROLS_DEFAULTS")

    if not hasattr(module, "INJECTION_GUARD"):
        fail("contentblitz/config.py must define INJECTION_GUARD")

    ok("config.py defines RETRY_POLICY, COST_CONTROLS_DEFAULTS, and INJECTION_GUARD")


def validate_state() -> None:
    module = load_module("contentblitz/state.py", "contentblitz_state")

    if not hasattr(module, "ContentBlitzState"):
        fail("contentblitz/state.py must define ContentBlitzState")

    state = module.ContentBlitzState

    annotations = getattr(state, "__annotations__", {})
    missing = STATE_TOP_LEVEL_FIELDS - set(annotations.keys())

    if missing:
        fail(
            "ContentBlitzState missing required fields:\n"
            + "\n".join(f"  - {f}" for f in sorted(missing))
        )

    ok("ContentBlitzState includes all required top-level fields")


def validate_workflow_files() -> None:
    required = {
        "contentblitz/workflow/graph.py",
        "contentblitz/workflow/routing.py",
    }

    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        fail("Missing workflow files:\n" + "\n".join(f"  - {p}" for p in missing))

    ok("Workflow files exist")


def validate_no_external_calls_in_phase1() -> None:
    forbidden_patterns = [
        "openai.OpenAI(",
        "requests.get(",
        "requests.post(",
        "client.chat.completions.create",
        "client.images.generate",
    ]

    offenders = []
    for path in (ROOT / "contentblitz").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                offenders.append((str(path.relative_to(ROOT)), pattern))

    if offenders:
        warn(
            "Potential real API calls found. "
            "Acceptable only inside mocked/stub tool modules for later phases:"
        )
        for file, pattern in offenders:
            print(f"  - {file}: {pattern}")
    else:
        ok("No obvious real external API calls detected")


def validate_spec_present() -> None:
    spec_path = ROOT / "docs" / "ContentBlitz_Execution_Spec.md"
    if not spec_path.exists():
        fail("docs/ContentBlitz_Execution_Spec.md is missing")

    text = spec_path.read_text(encoding="utf-8")
    required_phrases = [
        "ContentBlitz",
        "DEVELOPMENT ENVIRONMENT",
        "pytest",
        "RETRY_POLICY",
        "Consistency Enforcement Rules",
    ]

    missing = [p for p in required_phrases if p not in text]
    if missing:
        fail(
            "Spec file exists but appears incomplete. Missing phrases:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    ok("Execution spec exists and contains required sections")


def validate_authoritative_nodes() -> None:
    expected_nodes_from_modules = set(EXPECTED_AGENT_MODULES.values())

    if AUTHORITATIVE_NODES != expected_nodes_from_modules:
        fail(
            "AUTHORITATIVE_NODES does not match EXPECTED_AGENT_MODULES values.\n"
            f"AUTHORITATIVE_NODES:      {sorted(AUTHORITATIVE_NODES)}\n"
            f"EXPECTED_AGENT_MODULES:   {sorted(expected_nodes_from_modules)}"
        )

    if len(AUTHORITATIVE_NODES) != 12:
        fail(
            "AUTHORITATIVE_NODES must contain exactly 12 nodes, "
            f"found {len(AUTHORITATIVE_NODES)}"
        )

    ok("AUTHORITATIVE_NODES matches EXPECTED_AGENT_MODULES and contains 12 nodes")


def validate_agent_exports() -> None:
    missing_exports = []

    for filename, function_name in EXPECTED_AGENT_MODULES.items():
        module_path = f"contentblitz/agents/{filename}"
        module_name = f"contentblitz.agents.{filename.removesuffix('.py')}"

        module = load_module(module_path, module_name)

        if not hasattr(module, function_name):
            missing_exports.append(f"{module_path} must export {function_name}()")

    if missing_exports:
        fail(
            "Missing required agent node function exports:\n"
            + "\n".join(f"  - {item}" for item in missing_exports)
        )

    ok("All agent modules export required node functions")


def validate_agent_files() -> None:
    expected_files = set(EXPECTED_AGENT_MODULES.keys()) | {"__init__.py"}

    agents_dir = ROOT / "contentblitz" / "agents"
    found = {p.name for p in agents_dir.glob("*.py")}
    missing = expected_files - found

    if missing:
        fail(
            "Missing required agent module files:\n"
            + "\n".join(f"  - contentblitz/agents/{m}" for m in sorted(missing))
        )

    ok("All required agent module files exist")


def main() -> None:
    print("\nContentBlitz Phase 1 Structural Validation\n" + "-" * 45)

    require_paths()
    validate_spec_present()
    validate_requirements()
    validate_gitignore()
    validate_no_unittest()
    validate_config()
    validate_state()
    validate_agent_files()
    validate_authoritative_nodes()
    validate_agent_exports()
    validate_workflow_files()
    validate_no_external_calls_in_phase1()

    print("\n✅ Phase 1 structural validation passed.\n")


if __name__ == "__main__":
    main()
