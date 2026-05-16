from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


LIVE_TEST_FILES = [
    "tests/live/test_live_generate_text.py",
    "tests/live/test_live_search_web.py",
    "tests/live/test_live_generate_image.py",
    "tests/live/test_live_openai_agent_integration.py",
]

MANUAL_SCRIPT_FILES = [
    "scripts/dev/manual_live_openai_agent.py",
    "scripts/dev/manual_live_web_search.py",
    "scripts/dev/manual_cache_check.py",
    "scripts/dev/smoke_phase2_live.py",
]

REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "SERP_API_KEY",
    "PERPLEXITY_API_KEY",
]


def _exists(path: str) -> bool:
    return (PROJECT_ROOT / path).exists()


def _print_file_inventory(title: str, files: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for rel_path in files:
        status = "FOUND" if _exists(rel_path) else "MISSING"
        print(f"[{status}] {rel_path}")


def _print_env_summary() -> None:
    live_enabled = os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") == "1"
    live_image_enabled = os.getenv("CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS") == "1"

    print("\nEnvironment Summary")
    print("-------------------")
    print(f"CONTENTBLITZ_RUN_LIVE_TESTS enabled: {live_enabled}")
    print(f"CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS enabled: {live_image_enabled}")
    for key in REQUIRED_ENV_VARS:
        print(f"{key} present: {bool(os.getenv(key))}")


def _print_commands() -> None:
    print("\nManual Commands")
    print("---------------")
    print(
        "pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing"
    )
    print("pytest tests/live -rs")
    print("python scripts/dev/smoke_phase2_live.py --dry-run")
    print(
        "CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_generate_text.py -s -rs"
    )
    print(
        "CONTENTBLITZ_RUN_LIVE_TESTS=1 pytest tests/live/test_live_search_web.py -s -rs"
    )
    print(
        "CONTENTBLITZ_RUN_LIVE_TESTS=1 CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS=1 "
        "pytest tests/live/test_live_generate_image.py -s -rs"
    )


def _run_pytest(test_targets: list[str]) -> int:
    command = [sys.executable, "-m", "pytest", *test_targets, "-s", "-rs"]
    print("\nRunning:", " ".join(command))
    try:
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    except Exception as exc:
        print(f"Command failed safely: {exc.__class__.__name__}: {exc}")
        return 1
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and optionally run Phase 2 live smoke tests."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report configuration only. Makes no provider API calls.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute live pytest smoke tests when flags are enabled.",
    )
    args = parser.parse_args()

    print("ContentBlitz Phase 2 Live Smoke Audit")
    print("=====================================")
    _print_file_inventory("Live Test Files", LIVE_TEST_FILES)
    _print_file_inventory("Manual Script Files", MANUAL_SCRIPT_FILES)
    _print_env_summary()
    _print_commands()

    if args.dry_run:
        print("\nDry-run mode: no API calls were made.")
        return 0

    if not args.run:
        print("\nNo live execution requested. Use --run to execute live smoke tests.")
        return 0

    if os.getenv("CONTENTBLITZ_RUN_LIVE_TESTS") != "1":
        print("\nLive execution blocked: CONTENTBLITZ_RUN_LIVE_TESTS is not enabled.")
        return 0

    status = 0
    status |= _run_pytest(
        [
            "tests/live/test_live_generate_text.py",
            "tests/live/test_live_search_web.py",
        ]
    )

    if os.getenv("CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS") == "1":
        status |= _run_pytest(["tests/live/test_live_generate_image.py"])
    else:
        print(
            "\nSkipping live image execution: CONTENTBLITZ_RUN_LIVE_IMAGE_TESTS is not enabled."
        )

    return 0 if status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
