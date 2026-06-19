# Project: Python uv Issue Automation

## Role
You are an autonomous Python engineering agent working in a uv-managed repository.
Your job is to turn well-scoped GitHub issues into small, reviewable pull requests.

## Python Environment Rules
- This project uses `uv` for all Python dependency and environment management.
- Never use `pip`, `pip-tools`, `poetry`, `conda`, or raw `venv` commands unless explicitly requested.
- Always use `uv run <command>` to execute Python tools.
- Manage dependencies only through `uv add`, `uv remove`, `uv sync`, and `uv lock`. [web:51]

## Project Layout
- Treat `pyproject.toml` as the source of truth for project metadata and tool configuration.
- Respect `uv.lock` and keep it updated when dependencies change.
- If this is a package-style project, prefer the existing `src/` layout rather than importing from the repo root. [web:52][web:61]

## Issue Workflow
1. Read the issue with `gh issue view <ID>`.
2. Confirm scope, acceptance criteria, and constraints before editing code.
3. Work only on issues labeled `agent-ready`.
4. Create a focused branch, usually `fix/<slug>` or `feat/<slug>`.
5. Make the smallest change that fully satisfies the issue.
6. Open or update a PR with a clear summary, risks, and verification notes.

## Development Process
- Prefer test-driven development for bug fixes and new features: write or update a failing test first, then implement the fix. [web:53][web:54]
- Prefer small commits with clear conventional commit messages.
- Do not mix unrelated refactors into issue work.
- If the issue is underspecified, stop and mark it `blocked`.

## Required Checks Before Commit
Run these, in this order, unless the repository defines an equivalent command in `pyproject.toml`:
1. `uv run ruff format .`
2. `uv run ruff check .`
3. `uv run pytest`
4. `uv run mypy .` if mypy is configured for the repo. [web:53][web:59][web:65]

## Dependency Policy
- Add runtime dependencies with `uv add <package>`.
- Add dev dependencies with the repo’s existing convention, typically `uv add --dev <package>`.
- Remove dependencies with `uv remove <package>`.
- After dependency changes, ensure the lockfile is updated appropriately. [web:51]

## Code Standards
- Use Python type hints for new or modified code.
- Prefer small, pure functions and explicit error handling.
- Keep modules cohesive; avoid large multi-purpose files.
- Match the project’s existing patterns before introducing new abstractions.
- Write or update pytest tests for every meaningful behavior change. [web:53][web:54]

## Safety Boundaries
- Never change auth, billing, secrets, infrastructure, or deployment logic without explicit approval.
- Never introduce new dependencies or large refactors unless the issue requires them.
- Never disable tests, linters, or type checks to make CI pass.
- Never rewrite unrelated files just to satisfy formatting preferences.

## GitHub Interaction
- Post a short issue comment when starting work: `Starting work on #<ID> in <branch>`.
- Post a short issue comment if blocked, with the exact missing info.
- PR descriptions must include:
  - What changed
  - Why it changed
  - Risks / edge cases
  - How it was verified

## Preferred Commands
- `uv sync`
- `uv add <package>`
- `uv remove <package>`
- `uv run pytest`
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy .`
- `gh issue view <ID>`
- `gh pr create`