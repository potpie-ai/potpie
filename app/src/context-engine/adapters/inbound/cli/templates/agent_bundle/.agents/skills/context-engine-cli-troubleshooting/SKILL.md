---
name: "context-engine-cli-troubleshooting"
description: "Use when the task is to diagnose why the context-engine CLI is failing, especially around `doctor`, Neo4j or Graphiti availability, env bootstrap, JSON output, or search/ingest runtime errors."
---

# Context-Engine CLI Troubleshooting

Use this skill for failures, broken setup, or unclear runtime behavior in the CLI.

## Load These Files First

- [`app/src/context-engine/adapters/inbound/cli/output.py`](app/src/context-engine/adapters/inbound/cli/output.py)
- [`app/src/context-engine/adapters/inbound/cli/main.py`](app/src/context-engine/adapters/inbound/cli/main.py)
- [`app/src/context-engine/adapters/outbound/settings_env.py`](app/src/context-engine/adapters/outbound/settings_env.py)
- [`app/src/context-engine/README.md`](app/src/context-engine/README.md)

## Failure Triage

1. Check whether `CONTEXT_GRAPH_ENABLED` is on.
2. Distinguish config presence from runtime availability:
   - `doctor` checks whether env values look set.
   - `search` / `ingest` instantiate `GraphitiEpisodicAdapter` and can fail on missing extras or Neo4j config.
3. For scope errors, switch to the pot-scope skill and inspect git/env resolution.
4. For formatting issues, verify both plain output and `--json`.

## Common Failure Buckets

- Context graph disabled.
- Neo4j env missing or pointed at the wrong source.
- Graphiti dependency unavailable.
- Pot inference failed because there is no active pot, no env map, or no readable git `origin`.
- Invalid ingest argument shape or invalid `--reference-time`.

## Editing Rules

- Keep stderr reserved for errors so stdout stays pipe-friendly.
- Preserve current exit-code semantics unless the task explicitly changes CLI contracts.
- When adding a new diagnostic, include a concrete hint for the next action.

## Verification

- `uv run context-engine doctor`
- `uv run context-engine --json doctor`
- `uv run pytest app/src/context-engine/tests/unit/`
