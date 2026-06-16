---
name: "potpie-pot-scope"
version: "3"
recommended: true
description: "Use when the task is about pot resolution: pot use, current-repo inference, registered repo sources, or graph/source command scope."
---

# Potpie Pot Scope

Use this skill when the hard part of the task is choosing or debugging the pot ID.

## Resolution Order

For inferred pot scope, the CLI follows this order:

1. Explicit `--pot <id-or-name>`.
2. Registered `repo` source matching the current working tree path or git
   `remote.origin.url`.
3. Active pot from `potpie pot use <id-or-name>`.
4. Fail with a clear error asking for setup, source registration, or `--pot`.

Pot inference picks the project pot. It does not automatically narrow reads to
the current repository. A pot can span multiple repos, so project timeline reads
should include all registered repo sources by default.

## Useful Diagnostics

```bash
potpie pot list          # show local pots and the active one
potpie pot info          # show active pot
potpie source list       # show sources registered to the active pot
```

## Setting Pot Scope

```bash
# Explicit active pot
potpie pot use pot_1ae8e967cd19

# Register the current repo path/source to the project pot
potpie source add repo /path/to/repo --pot <pot-id>
potpie source add repo github.com/owner/repo --pot <pot-id>
```

## Repo Parsing

Pot inference reads `git config --get remote.origin.url` and normalizes SSH/HTTPS
remotes to `host/owner/repo`. It also matches registered absolute repo paths.
When inference is ambiguous, pass `--pot` explicitly.

## Common Scoped Forms

```bash
# Inferred pot (from pot use / env / git)
potpie search "query"
potpie --json graph read --subgraph recent_changes --view timeline --limit 20

# Explicit pot
potpie search "query" --pot <pot-uuid>
potpie --json graph propose --file mutation.json --pot <pot-uuid>
potpie --json graph commit <plan_id> --pot <pot-uuid>
potpie source add repo <owner/repo> --pot <pot-uuid>
```
