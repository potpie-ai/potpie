---
name: "context-engine-pot-scope"
description: "Use when the task is about pot resolution for the context-engine CLI: `pot use`, env maps, git origin parsing, repo-to-pot mapping, `add`, or search/ingest scope inference."
---

# Context-Engine Pot Scope

Use this skill when the hard part of the task is choosing or debugging the pot ID.

## Load These Files First

- [`app/src/context-engine/adapters/inbound/cli/git_project.py`](app/src/context-engine/adapters/inbound/cli/git_project.py)
- [`app/src/context-engine/adapters/inbound/cli/credentials_store.py`](app/src/context-engine/adapters/inbound/cli/credentials_store.py)
- [`app/src/context-engine/adapters/inbound/cli/README.md`](app/src/context-engine/adapters/inbound/cli/README.md)

## Resolution Order

For inferred pot scope, follow the implementation exactly:

1. `context-engine pot use <uuid>` stored active pot.
2. `CONTEXT_ENGINE_REPO_TO_POT` match on `owner/repo`.
3. `CONTEXT_ENGINE_POTS` match on inverse map value.
4. Otherwise fail and tell the caller to pass a pot UUID explicitly or configure mappings.

## Repo Parsing Rules

- Read `git remote get-url origin`.
- Normalize SSH and HTTPS remotes to `owner/repo`.
- Keep matching case-insensitive on repo identifiers.
- `add` is diagnostic only; it does not persist a mapping.

## When Editing

- Keep the error messages actionable and explicit about the next valid commands.
- Preserve compatibility for both one-argument and two-argument `search` / `ingest` forms.
- If you change resolution semantics, update CLI README examples and help text in `main.py`.

## Useful Commands

- `uv run context-engine pot list`
- `uv run context-engine pot use <pot-uuid>`
- `uv run context-engine add .`
- `uv run context-engine search --cwd . "query"`
