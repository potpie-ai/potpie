---
name: "context-engine-cli"
description: "Use when the task is to run, explain, document, or modify the context-engine CLI in this repository. Covers command selection, flags, output modes, and the main code paths for `doctor`, `pot`, `add`, `search`, and `ingest`."
---

# Context-Engine CLI

Use this skill for work centered on the `context-engine` command in this repo.

## Load These Files First

- [`app/src/context-engine/adapters/inbound/cli/main.py`](app/src/context-engine/adapters/inbound/cli/main.py)
- [`app/src/context-engine/adapters/inbound/cli/README.md`](app/src/context-engine/adapters/inbound/cli/README.md)
- [`app/src/context-engine/adapters/inbound/cli/output.py`](app/src/context-engine/adapters/inbound/cli/output.py)

## Quick Workflow

1. Work from `app/src/context-engine` when running CLI commands.
2. Prefer `uv run context-engine ...` so the editable package and extras resolve consistently.
3. If the task changes command behavior, update both the Typer command in `main.py` and the CLI README examples.
4. Preserve both human output and `--json` output paths when editing behavior.

## Command Map

- `doctor`: reports whether key env/config values are present.
- `login` / `logout`: persist or clear Potpie API credentials in the user config dir.
- `pot use` / `pot unset` / `pot list` / `pot create`: manage local default pot state and inspect env-backed mappings.
- `add`: inspect a git checkout and print provider-scoped repo identity.
- `search`: semantic search over Graphiti episodic entities.
- `ingest`: write a raw episode into the episodic graph.

## Guardrails

- Do not describe the CLI as a separate production service; in Potpie it is an optional integration surface around the embedded package.
- `doctor` does not validate connectivity.
- `search` and `ingest` fail fast when `CONTEXT_GRAPH_ENABLED` is off or Graphiti is unavailable.
- Global `--source` is overridden by subcommand `--source`.

## Verification

- `uv run context-engine --json doctor`
- `uv run context-engine pot list`
- Add targeted unit tests under `app/src/context-engine/tests/unit/` when logic changes.
