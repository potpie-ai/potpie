---
name: potpie-cli
version: "5"
description: "Use when the task is centered on running, explaining, configuring, or troubleshooting the `potpie` command: doctor, login, pot management, source registration, search, graph workbench reads/writes, resource (document payload) commands, and pot scope behavior."
---

# Potpie CLI

Use this skill when the user is asking about the `potpie` command itself. For
ordinary project-memory context, prefer the relevant use-case skill first.

## Setup And Scope

For **repo-local** CLI install/reinstall from this checkout, use:

```bash
make cli-install    # UI build + stop old daemon + editable install
make cli-status
potpie doctor
```

Do not use raw `uv tool install --editable …` or `pip install` for day-to-day
local reinstalls. Reserve `uv tool install potpie` / `pip install potpie` for
the **published** package.

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
potpie pot default set --repo current <pot-id-or-alias>
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

`source add` accepts a closed set of kinds: `repo`, `linear`, `jira`,
`confluence`, `notion`, `url`. A repo on any host is `repo` — `github`,
`gitlab`, and `gitbucket` canonicalize to it. `--default` is repo-only.
A document is not a source: `source add pdf ./q3.pdf` exits 1 with
`source_kind_is_a_document`. Use the matching `potpie-resource-*` skill and
`potpie resource import` instead.

## Search

```bash
potpie search "query"
potpie --json search "query"
potpie search "query" --include decisions,features
potpie search "query" --intent debugging
potpie search "query" --pot <pot-id-or-alias>
```

The query is positional — there is no `--query`. Bare search resolves intent
`unknown`, which covers infra topology, timeline, decisions, and ingested
documents; `--intent` swaps in another task's families and `--include` names
families directly. `--help` lists both vocabularies, so neither has to be
guessed from subgraph names.

## Resources (document payloads)

```bash
potpie resource import <dir> --doc <slug> --source-ref <uri> --source-kind pdf
potpie resource get potpie://res/<doc>/<section>/0000 --with-neighbors
potpie resource list --doc <slug>
potpie resource rm <slug> --confirm
```

`import` absorbs a chunk directory an extraction script produced — atomic,
re-import replaces and bumps `revision`, and the document's structure is
written to the graph in the same command. `get` resolves chunk ids to text
with no graph query and batches multiple ids in one call. `rm` is destructive
and requires `--confirm`. `potpie doctor` reports resource store readiness.
Ingestion flows live in the per-format `potpie-resource-*` skills.

## Graph Workbench

```bash
potpie --json graph status
potpie --json graph catalog --task "<task>"
potpie --json graph describe <subgraph> --view <view> --examples
potpie graph read --subgraph <subgraph> --view <view> --limit 20
potpie graph search-entities "<name>" --type Service --limit 10
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
potpie --json graph quality summary
```

Use `potpie-graph` for advanced graph workbench details.

## Report Back

Answer with the command, not a description of it. `potpie doctor` said something
specific; paste the line you acted on, and paste the command that produced it.
Half of what this skill diagnoses is a wrong pot, a wrong host, or a stale
install, and every one of those is invisible in a summary and obvious in the
output the reader can see. Show the failing command *and* the fixing one when you
changed something.

Nothing in CLI troubleshooting is a shape, so no mermaid diagram here. A
pot/host/source state is a short list; write it as one.

## Boundaries

Repository links, docs, tickets, PRs, and logs are interpreted by the harness and
written with graph workbench mutations. Do not use pot-level connector queueing or
deterministic local code scans as the agent ingestion path.
Do not use scanner-driven graph updates.

For CLI failures, stay in this skill: run `potpie doctor`, `make cli-status`, or
`uv tool list` for install facts. For repo-local reinstall use `make cli-install`
(not raw `uv tool install`). Do not use `python -m pip show
potpie-context-engine` for local uv-tool installs. Inspect JSON output when
useful, check API URL/key config, confirm pot scope, and verify source
registration before changing project code.
