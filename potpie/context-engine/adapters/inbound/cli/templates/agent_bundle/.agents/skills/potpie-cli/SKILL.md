---
name: "potpie-cli"
description: "Use when the task involves running, explaining, or troubleshooting the Potpie CLI. Covers doctor, login, pot management, search, and ingest commands."
---

# Potpie CLI

Use this skill when the task is centered on the `potpie` command.

## Common Commands

```bash
# Setup and auth
potpie login <api-key> --url http://your-potpie-host
potpie doctor

# Pot management
potpie pot pots                   # list all pots
potpie pot use <uuid-or-alias>    # set active pot
potpie pot alias <name> <uuid>    # create local alias
potpie pot list                   # show env maps + active pot

# Search
potpie search "your query"
potpie search "query" -n 15
potpie search "query" --node-labels PullRequest,Decision
potpie search "query" --with-temporal

# Ingest
potpie ingest "Your note here"
potpie ingest --name "Title" --episode-body "Body" --source "meeting"
potpie ingest "Note" --sync
potpie ingest --file notes.txt

# Machine-readable output
potpie --json doctor
potpie --json search "query"
potpie --json ingest "note"
```

## Setup Flow

1. `potpie login <key> --url <host>` — save credentials.
2. `potpie doctor` — verify health + auth.
3. `potpie pot pots` — list available pots.
4. `potpie pot use <id>` — set active pot.

## How Pot Scope Is Resolved

For `search` and `ingest`, the pot is chosen in this order:
1. Explicit UUID passed as first argument.
2. Active pot from `pot use`.
3. `CONTEXT_ENGINE_REPO_TO_POT` env map matched on `owner/repo`.
4. `CONTEXT_ENGINE_POTS` env map (inverse lookup).
5. Git `origin` remote parsed from `--cwd`.
6. Fail with a clear error.

## Key Flags

- `--json` — machine-readable JSON on stdout (global, before subcommand).
- `--verbose` / `-v` — full tracebacks on errors.
- `--source` — default source label for ingest (global) or search filter.
- `--sync` — synchronous ingest (wait for apply, not just queue).
- `--cwd` — git working tree for pot inference.

## Common Failures

| Symptom | Fix |
|---------|-----|
| `Potpie API not configured` | Set `POTPIE_API_URL` + `POTPIE_API_KEY`, or run `potpie login`. |
| `401 Invalid API key` | Key is wrong or expired. Re-run `potpie login`. |
| `Pot scope required` | Run `potpie pot use <id>` or pass a pot UUID explicitly. |
| `GET /health` fails | Wrong base URL, wrong port, or server is down. |
| `409 duplicate_ingest` | Same idempotency key already ingested. Safe to ignore or use `--idempotency-key`. |
