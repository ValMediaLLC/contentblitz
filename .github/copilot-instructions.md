Review this project against ContentBlitz architecture rules, ContentBlitz_Execution_Spec.md and pre-commit validation requirements. If any of the blocking issues are present, flag them as critical and do not proceed with implementation until they are resolved. If all blocking issues are resolved, ensure that all required validation steps are completed successfully before committing the implementation step.

Blocking issues:
- Any change to the 12 authoritative nodes
- Non-deterministic routing
- Routing function returning None
- Tools mutating state
- retry_counts incremented outside retry_router
- Real provider calls in unit/integration tests
- API keys, secrets, .env, stack traces, or base64 image data committed
- output_assembler/export/clarification mutating drafts

Required validation:
- python scripts/validate_phase1.py
- python scripts/validate_phase2.py
- python scripts/validate_phase3.py
- pytest tests/unit tests/integration --cov=contentblitz --cov-report=term-missing

Ruff validation:
- Follow pyproject.toml Ruff configuration.
- Do not run broad auto-formatting during feature/bugfix commits.
- Only modify files required for the requested change.
- If formatting is needed, limit it to touched files.
- Do not apply repo-wide ruff --fix or ruff format unless explicitly asked.