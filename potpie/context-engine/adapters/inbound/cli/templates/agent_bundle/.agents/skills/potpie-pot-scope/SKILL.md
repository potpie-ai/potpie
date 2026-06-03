---
name: "potpie-pot-scope"
description: "Use when the task is about pot resolution: pot use, env maps, git origin parsing, repo-to-pot mapping, or search/ingest scope inference."
---

# Potpie Pot Scope

Use this skill when the hard part of the task is choosing or debugging the pot ID.

## Resolution Order

For inferred pot scope, the CLI follows this order:

1. `potpie pot use <uuid>` — stored active pot (highest priority).
2. `CONTEXT_ENGINE_REPO_TO_POT` env map — matched on `owner/repo`.
3. `CONTEXT_ENGINE_POTS` env map — inverse map lookup.
4. Git `origin` remote parsed from `--cwd` — matched against env maps.
5. Fail with a clear error asking for an explicit pot UUID.

## Useful Diagnostics

```bash
potpie pot list          # show active pot, env maps, local aliases
context-engine add .             # show parsed git remote + active pot for cwd
potpie pot pots          # list server-side pots you have access to
```

## Setting Pot Scope

```bash
# Explicit UUID
potpie pot use aeaa1c74-2a0e-4b7c-ab03-1da1390d32b6

# With a local alias (easier to type)
potpie pot alias myproject aeaa1c74-2a0e-4b7c-ab03-1da1390d32b6
potpie pot use myproject

# Via env (for CI or multi-repo setups)
export CONTEXT_ENGINE_REPO_TO_POT='{"owner/repo":"pot-uuid"}'
```

## Repo Parsing

`potpie add .` reads `git remote get-url origin` and normalizes both SSH and HTTPS remotes to `owner/repo`. Use it to check what the CLI sees for the current directory.

## Multi-Argument Forms

`search` and `ingest` accept an optional first-argument pot UUID:

```bash
# Inferred pot (from pot use / env / git)
potpie search "query"
potpie ingest "episode body"

# Explicit pot UUID
potpie search <pot-uuid> "query"
potpie ingest <pot-uuid> "episode body"
```
