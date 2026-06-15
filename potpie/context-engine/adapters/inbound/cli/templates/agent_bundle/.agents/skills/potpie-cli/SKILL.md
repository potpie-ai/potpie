---
name: "potpie-cli"
version: "3"
recommended: true
description: "Use when the task involves running, explaining, or troubleshooting the Potpie CLI. Covers doctor, login, pot management, source registration, search, and graph read/write commands."
---

# Potpie CLI

Use this skill when the task is centered on the `potpie` command.

## Common Commands

```bash
# Setup and auth
potpie doctor

# Pot management
potpie pot list                   # list local pots
potpie pot use <uuid-or-alias>    # set active pot
potpie pot info                   # show active pot
potpie pot create <name> --repo <owner/repo> --use

# Source registration, not ingestion
potpie source add repo .                 # register the current repo (resolved to remote/abs path)
potpie source add repo <owner/repo> --pot <pot>
potpie source list --pot <pot>

# Context search
potpie search "your query"
potpie search "query" -n 15
potpie search "query" --node-labels PullRequest,Decision
potpie search "query" --with-temporal

# Graph memory
potpie --json graph catalog
potpie --json graph read --view preferences.active_preferences --scope repo:<owner-repo>
potpie --json timeline recent --limit 20
potpie --json graph read --view recent_changes.timeline --time-window 7d --limit 20
potpie --json graph search-entities "payments api" --type Service
potpie graph mutation-template --kind repo-baseline   # schema-only mutation skeleton
potpie --json graph mutate --file mutation.json --dry-run

# Machine-readable output
potpie --json doctor
potpie --json search "query"
potpie --json graph catalog
```

## Setup Flow

1. `potpie login <key> --url <host>` — save credentials.
2. `potpie doctor` — verify health + auth.
3. `potpie pot list` — list local pots.
4. `potpie pot use <id>` — set active pot.

## How Pot Scope Is Resolved

For graph, search, source, and pot-scoped commands, the pot is chosen in this order:
1. Explicit `--pot` id/name.
2. A registered `repo` source matching the current working tree path or git
   `remote.origin.url`.
3. Active pot from `pot use`.
4. Fail with a clear error.

Pot inference chooses the project pot only. It does not automatically narrow a
timeline read to the current repo, because a pot can span multiple repositories.
Use `potpie timeline recent` for project-wide history across repos; pass
`--service` or `graph read --scope ...` only when the user asks for a narrower
slice.

## Key Flags

- `--json` — machine-readable JSON on stdout (global, before subcommand).
- `--verbose` / `-v` — full tracebacks on errors.
- `--pot` — explicit pot id/name on commands that accept it.
- `graph read --since`, `--until`, `--time-window` — temporal bounds.
- `graph read --format events|raw|jsonl` — timeline defaults to deduped events.

## Ingestion Boundary

There is no local code scan command and no generic note-ingest command in the
agent path. Repository links, docs, tickets, and PRs are interpreted by the
harness, then written with `potpie graph mutate` or `context_record`.

## Common Failures

| Symptom | Fix |
|---------|-----|
| `Potpie API not configured` | Set `POTPIE_API_URL` + `POTPIE_API_KEY`, or run `potpie login`. |
| `401 Invalid API key` | Key is wrong or expired. Re-run `potpie login`. |
| `Pot scope required` | Run `potpie pot use <id>` or pass a pot UUID explicitly. |
| `GET /health` fails | Wrong base URL, wrong port, or server is down. |
| `invalid_mutation_payload` | Validate the JSON shape with `potpie --json graph catalog` and retry `graph mutate --dry-run`. |
