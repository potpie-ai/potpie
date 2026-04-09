# Potpie Agent Guide

## Scope

This repository contains the Potpie API and an embedded `context-engine` package. When a task mentions the `context-engine` CLI, work from the package sources under [`app/src/context-engine`](app/src/context-engine).

## Context-Engine CLI

- Entry point: [`app/src/context-engine/adapters/inbound/cli/main.py`](app/src/context-engine/adapters/inbound/cli/main.py)
- CLI docs: [`app/src/context-engine/adapters/inbound/cli/README.md`](app/src/context-engine/adapters/inbound/cli/README.md)
- Package overview: [`app/src/context-engine/README.md`](app/src/context-engine/README.md)

Use the repo-local skills under `.agents/skills/` when the task is about:

- running or explaining CLI commands
- resolving pot scope from git remotes or env maps
- debugging `doctor`, `search`, `ingest`, or Neo4j/Graphiti setup

## Working Rules

- Prefer `uv run context-engine ...` from [`app/src/context-engine`](app/src/context-engine).
- For pot inference, follow the code path in [`app/src/context-engine/adapters/inbound/cli/git_project.py`](app/src/context-engine/adapters/inbound/cli/git_project.py): active pot from `pot use`, then env maps, then git `origin`, else fail.
- `search` and `ingest` require `CONTEXT_GRAPH_ENABLED` plus working Neo4j/Graphiti config. `doctor` only reports configuration presence, not connectivity.
- Keep changes aligned with Typer patterns already used in the CLI and update the CLI README when behavior changes.

## Validation

- Focused CLI tests live under [`app/src/context-engine/tests`](app/src/context-engine/tests).
- Common verification:
  - `uv run pytest app/src/context-engine/tests/unit/`
  - `uv run context-engine doctor`
