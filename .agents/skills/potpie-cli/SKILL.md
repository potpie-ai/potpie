---
name: "potpie-cli"
description: "Use when the task is to run, explain, document, or modify the Potpie CLI in this repository. Covers command selection, flags, output modes, and the main code paths for `doctor`, `pot`, `add`, `search`, and `ingest`."
---

# Potpie CLI

Use this skill for work centered on the `potpie` command in this repo.

## Load These Files First

- [`app/src/context-engine/adapters/inbound/cli/main.py`](app/src/context-engine/adapters/inbound/cli/main.py)
- [`app/src/context-engine/adapters/inbound/cli/README.md`](app/src/context-engine/adapters/inbound/cli/README.md)
- [`app/src/context-engine/adapters/inbound/cli/output.py`](app/src/context-engine/adapters/inbound/cli/output.py)

## Quick Workflow

1. Work from `app/src/context-engine` when running CLI commands.
2. Prefer `uv run potpie ...` so the editable package and extras resolve consistently.
3. If the task changes command behavior, update both the Typer command in `main.py` and the CLI README examples.
4. Preserve both human output and `--json` output paths when editing behavior.

## Command Map

- `doctor`: Potpie API URL/key hints + optional `GET /health` when configured.
- `login` / `logout`: persist or clear Potpie API credentials in the user config dir.
- `pot create` / `pot pots` (server-owned context pots), `pot repo list` / `pot repo add`, `pot alias` / `pot use` / `pot unset` / `pot list` / `pot clear-local`; `logout` clears stored credentials.
- `add`: inspect a git checkout and print provider-scoped repo identity.
- `search` / `ingest` / `pot hard-reset`: HTTP to Potpie **`POST /api/v2/context/*`** with **`X-API-Key`** (`POTPIE_API_URL` + `POTPIE_API_KEY` or `potpie login`).

## Guardrails

- Do not describe the CLI as a separate production service; in Potpie it is an optional integration surface around the embedded package.
- `doctor` probes **`GET /health`** when base URL and API key resolve; it does not validate graph content.
- `search` and `ingest` require a reachable Potpie API and valid key (no local Neo4j/Graphiti required on the CLI host).
- Global `--source` is overridden by subcommand `--source`.

## Verification

- `uv run potpie --json doctor`
- `uv run potpie pot list`
- Add targeted unit tests under `app/src/context-engine/tests/unit/` when logic changes.
