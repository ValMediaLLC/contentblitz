#!/usr/bin/env python3
"""Optional LangSmith smoke script with safe dry-run defaults."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Sequence

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - defensive import guard
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contentblitz.core.observability import (  # noqa: E402
    get_workflow_tracer,
    safe_workflow_end_metadata,
    safe_workflow_start_metadata,
)
from contentblitz.core.redaction import normalize_trace_error  # noqa: E402
from contentblitz.ui.observability import (  # noqa: E402
    build_observability_diagnostics,
)

_GATE_ENV = "CONTENTBLITZ_RUN_LANGSMITH_SMOKE"
_REQUIRED_ENV_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    _GATE_ENV,
)


def _maybe_load_dotenv() -> None:
    if load_dotenv is None:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_presence() -> dict[str, bool]:
    return {key: bool(os.getenv(key)) for key in _REQUIRED_ENV_VARS}


def _print_env_presence() -> None:
    print("\nEnvironment Presence")
    print("--------------------")
    for key, present in _env_presence().items():
        print(f"{key} present: {present}")


def _live_smoke_allowed() -> bool:
    return os.getenv(_GATE_ENV) == "1"


def _attempt_live_trace() -> tuple[bool, str]:
    session_id = f"phase4-smoke-{uuid.uuid4().hex[:8]}"
    initial_state = {
        "session_id": session_id,
        "requested_outputs": ["blog"],
        "workflow_status": "running",
        "routing_decision": "query_handler_node",
    }
    final_state = {"workflow_status": "success"}
    try:
        tracer = get_workflow_tracer()
        span = tracer.start_workflow(
            metadata=safe_workflow_start_metadata(initial_state)
        )
        span.finish(
            metadata=safe_workflow_end_metadata(
                initial_state=initial_state,
                final_state=final_state,
            ),
            outputs={"workflow_status": "success"},
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        safe_error = normalize_trace_error(exc)
        return False, str(safe_error.get("message", "Trace submission failed."))
    return True, f"Trace attempt submitted (session_id={session_id})."


def main(argv: Sequence[str] | None = None) -> int:
    _maybe_load_dotenv()
    parser = argparse.ArgumentParser(
        description="Safe LangSmith smoke script (dry-run by default)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report safe diagnostics only. Makes no LangSmith calls.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    diagnostics = build_observability_diagnostics()
    print("ContentBlitz LangSmith Smoke")
    print("============================")
    print(f"Observability status: {diagnostics['status_label']}")
    print(f"Tracing enabled: {bool(diagnostics['tracing_enabled'])}")
    print(f"Project: {diagnostics['project_name']}")
    print(f"Endpoint host: {diagnostics['endpoint_host']}")
    print(f"Last trace attempt: {diagnostics['last_trace_attempt_label']}")
    _print_env_presence()

    if args.dry_run:
        print("\nDry-run mode complete. No LangSmith calls were made.")
        return 0

    if not _live_smoke_allowed():
        print(
            (
                f"\nLive smoke skipped. Set {_GATE_ENV}=1 to permit a live trace "
                "attempt."
            )
        )
        return 0

    if not bool(diagnostics.get("tracing_enabled", False)):
        print(
            (
                "\nLive smoke not attempted: tracing is unavailable. "
                "Confirm LANGSMITH_TRACING and LANGSMITH_API_KEY."
            )
        )
        return 0

    success, summary = _attempt_live_trace()
    print(f"\nLive smoke result: {'success' if success else 'failed'}")
    print(f"Summary: {summary}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
