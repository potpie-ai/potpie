---
name: potpie-cli
version: "2"
description: "Use when the task is centered on running, explaining, configuring, or troubleshooting the `potpie` command: doctor, login, pot management, source registration, search, graph workbench reads/writes, and pot scope behavior."
---

# Potpie CLI

Use this skill when the user is asking about the `potpie` command itself. For
ordinary project-memory context, prefer the relevant use-case skill first.

## Setup And Scope

```bash
potpie doctor
potpie --json doctor
make cli-status
uv tool list
which -a potpie
potpie login <api-key> --url <host>
potpie pot list
potpie pot use <pot-id-or-alias>
potpie --json pot info
potpie pot linked --repo current
potpie pot default --repo current --set <pot-id-or-alias>
potpie --json source list
potpie source add repo .
potpie source add repo <owner/repo> --pot <pot>
```

Pot scope resolves in this order:

1. Explicit `--pot`.
2. Repo-local default set by `source add repo` or `pot default set`.
3. Registered repo source matching the current working tree path or
   `remote.origin.url`.
4. Active pot from `potpie pot use`.
5. Clear failure asking for setup, source registration, default selection, or
   explicit `--pot`.

A pot is a project boundary and may span multiple repos. Do not automatically
narrow timeline reads to the current repo.

`source add repo` sets the repo-local default by default. Use `--no-default`
only when deliberately registering a repo to a non-default pot. If graph output
warns that the selected pot is empty but another linked pot has claims, run the
suggested `pot default set --repo current <pot>` command before continuing.

## Search

```bash
potpie search "query"
potpie --json search "query"
potpie search "query" --include decisions,features
potpie search "query" --pot <pot-id-or-alias>
```

## Graph Workbench

```bash
potpie --json graph status
potpie --json graph catalog --task "<task>"
potpie --json graph catalog --subgraph <subgraph>
potpie graph read --subgraph <subgraph> --view <view> --limit 20
potpie graph search-entities "<name>" --type Service --limit 10
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
potpie --json graph quality summary
```

Use `potpie-graph` for advanced graph workbench details.

## Boundaries

Repository links, docs, tickets, PRs, and logs are interpreted by the harness and
written with graph workbench mutations. Do not use pot-level connector queueing or
deterministic local code scans as the agent ingestion path.
Do not use scanner-driven graph updates.

For CLI failures, stay in this skill: run `potpie doctor`, `make cli-status`, or
`uv tool list` for install facts. Do not use `python -m pip show
potpie-context-engine` for local uv-tool installs. Inspect JSON output when
useful, check API URL/key config, confirm pot scope, and verify source
registration before changing project code.
