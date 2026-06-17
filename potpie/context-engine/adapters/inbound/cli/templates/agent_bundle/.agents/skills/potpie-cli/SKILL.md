---
name: potpie-cli
description: "Use when the task is centered on running, explaining, configuring, or troubleshooting the `potpie` command: doctor, login, pot management, source registration, search, graph workbench reads/writes, and pot scope behavior."
---

# Potpie CLI

Use this skill when the user is asking about the `potpie` command itself. For
ordinary project-memory context, prefer the relevant use-case skill first.

## Setup And Scope

```bash
potpie doctor
potpie --json doctor
potpie login <api-key> --url <host>
potpie pot list
potpie pot use <pot-id-or-alias>
potpie --json pot info
potpie --json source list
potpie source add repo .
potpie source add repo <owner/repo> --pot <pot>
```

Pot scope resolves in this order:

1. Explicit `--pot`.
2. Registered repo source matching the current working tree path or
   `remote.origin.url`.
3. Active pot from `potpie pot use`.
4. Clear failure asking for setup, source registration, or explicit `--pot`.

A pot is a project boundary and may span multiple repos. Do not automatically
narrow timeline reads to the current repo.

## Search

```bash
potpie search "query"
potpie --json search "query" -n 15
potpie search "query" --node-labels PullRequest,Decision
potpie search "query" --with-temporal
```

## Graph Workbench

```bash
potpie --json graph status
potpie --json graph catalog --task "<task>"
potpie --json graph describe <subgraph> --view <view> --examples
potpie --json graph read --subgraph <subgraph> --view <view> --limit 20
potpie --json graph search-entities "<name>" --type Service --limit 10
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id>
potpie --json graph history --plan <plan_id>
potpie --json graph quality summary
```

Use `potpie-graph` for advanced graph workbench details.

## Boundaries

Repository links, docs, tickets, PRs, and logs are interpreted by the harness and
written with graph workbench mutations. Do not use pot-level connector queueing or
deterministic local code scans as the agent ingestion path.
Do not use scanner-driven graph updates.

For CLI failures, stay in this skill: run `potpie doctor`, inspect JSON output
when useful, check API URL/key config, confirm pot scope, and verify source
registration before changing project code.
