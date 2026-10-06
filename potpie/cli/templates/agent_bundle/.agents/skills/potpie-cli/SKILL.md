---
name: potpie-cli
version: "4"
description: "Use when the task is centered on running, explaining, configuring, or troubleshooting the `potpie` command: doctor, login, pot management, source registration, resolve/search/record, graph workbench reads/writes, and pot scope behavior."
---

# Potpie CLI

Use this skill when the user is asking about the `potpie` command itself. For
ordinary project-memory context, prefer the relevant use-case skill first.

## Setup And Scope

Install the **published** package with `uv tool install potpie` or
`pip install potpie`. Only inside the potpie source checkout — the repo whose
`Makefile` has a `cli-install` target — reinstall from source with:

```bash
make cli-install    # UI build + stop old daemon + editable install
make cli-status
potpie doctor
```

Anywhere else there is no Makefile and `make cli-status` is a dead call; the
install facts come from `uv tool list` and `which -a potpie`. Do not use raw
`uv tool install --editable …` or `pip install` for day-to-day reinstalls of
the checkout. If `potpie backend doctor` names a missing driver, that is a
packaging answer — report what it names; do not treat it as a broken graph.

```bash
potpie status
potpie doctor
potpie --json doctor
uv tool list
which -a potpie
potpie login --api-key <key> --url <host>
potpie pot list
potpie pot use <pot-id-or-alias>
potpie --json pot info
potpie pot linked --repo current
potpie pot default set --repo current <pot-id-or-alias>
potpie --json source list
potpie source add repo .
potpie source add repo <owner/repo> --pot <pot>
```

`login` takes flags, not a positional key. `potpie status` is the one health
check: daemon, pot, backend readiness, claim counts and open quality findings
in a few lines; `doctor` adds the install, the backend's capabilities and the
repo → pot mapping.

Pot scope for `graph …`, `resolve`, `search` and `record` resolves in this
order:

1. Explicit `--pot`.
2. Repo-local default set by `source add repo` or `pot default set`.
3. Registered repo source matching the current working tree path or
   `remote.origin.url`.
4. Active pot from `potpie pot use`.
5. Clear failure asking for setup, source registration, default selection, or
   explicit `--pot`.

`potpie status` reports the *active* pot (step 4) rather than the repo default,
so when the two differ, trust the header of a read or `doctor` for the pot a
read will hit. Use `status --harness codex` (or your harness) for skill
readiness; setup/skills use `--agent`, while status uses `--harness`. `doctor`
reports the current routing and does not accept `--pot`. A `--pot` value is a
pot id or name.

A pot is a project boundary and may span multiple repos. Do not automatically
narrow timeline reads to the current repo.

`source add repo` sets the repo-local default by default. Use `--no-default`
only when deliberately registering a repo to a non-default pot. If graph output
warns that the selected pot is empty but another linked pot has claims, run the
suggested `pot default set --repo current <pot>` command before continuing.
`source add <kind> <location>` registers metadata only — it never ingests or
scans; for a repo, `.` or `current` registers the checkout you are in.

## Context Verbs

```bash
potpie status
potpie resolve "<task>"
potpie resolve "<task>" --intent debugging --include prior_bugs,docs,timeline
potpie search "query"
potpie --json search "query"
potpie search "query" --include decisions,features
potpie record --type decision --summary "<the decision>" --detail rationale="<why>" --scope service:<name>
```

Use one shared discovery pass across skills: run `resolve` concurrently with
scope-only `preferences_for_scope --repo current` for code work and untyped
`graph search-entities` for explicitly named entities whose keys are unknown.
Reuse current results and hook context for the same task, pot, and scope. Run
`status` alongside discovery if health needs checking; resolve ambiguous pot
routing first and check pot IDs before combining results. Use returned keys for
focused views; those follow-ups can run concurrently once inputs are known.
Stop when evidence and constraints are covered. Load the relevant use-case
skills together; do not repeat resolve for each skill.

`resolve` is the broad discovery read: the intent is inferred from the task
text when `--intent` is omitted, and the reply is a bounded envelope of
`subject PREDICATE object · fact` rows across families with a `+N more` footer.
`search` is the follow-up for a known phrase; its query is positional — there
is no `--query`. Bare search is broad (`--include docs` narrows it to document
sections) and infers the definition intent for acronym questions. Unknown
include names are errors. `--help` lists both vocabularies. `confidence` in
either header is coverage, not a verdict; a small pot reads `low` with the
right rows on top. `record` is the one-call write for a fix, decision,
preference, bug pattern or verification; `--type` help names the `--detail`
keys each type requires, and a repeated `--detail` key builds a list. A
free-form note (`workflow`, `service_note`, `runbook_note`, …) takes a summary
and any `--detail`.

## Graph Workbench

```bash
potpie graph catalog
potpie graph describe <subgraph> --view <view>
potpie graph read --subgraph <subgraph> --view <view> --limit 20
potpie graph search-entities "<name>" --limit 10
potpie graph mutation-template --kind <kind>
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph quality summary
```

One rule for `--json`: text for reads, `--json` for `propose`, `commit`, and
anything a script parses. `catalog --task` does not narrow the catalog. Before
ingestion, `graph catalog --profile full` exposes all public entity types with
their identity policy and descriptions, predicates, and allowed endpoints;
`--profile read`
stays compact for read-view discovery. Use the full ontology and the selection
guidance in `potpie-graph` before choosing a write shape. `describe --examples`
prints its examples only with `--json` and carries read examples only; the
write payload shape is `mutation-template`. `commit --verify` prints the plan
id, readback and quality status, so `graph history --plan <plan_id>` is for
later inspection.

Use `potpie-graph` for advanced graph workbench details.

## Report Back

Answer with the command, not a description of it. `potpie doctor` said something
specific; paste the line you acted on, and paste the command that produced it.
Half of what this skill diagnoses is a wrong pot, a wrong backend, or a stale
install, and every one of those is invisible in a summary and obvious in the
output the reader can see. Show the failing command *and* the fixing one when you
changed something.

Nothing in CLI troubleshooting is a shape, so no mermaid diagram here. A
pot/source/backend state is a short list; write it as one.

## Boundaries

Repository links, docs, tickets, PRs, and logs are interpreted by the harness and
written with `potpie record` or graph workbench mutations. Do not use pot-level
connector queueing or deterministic local code scans as the agent ingestion path.
Do not use scanner-driven graph updates.

For CLI failures, stay in this skill: run `potpie status`, `potpie doctor`, or
`uv tool list` for install facts (`make cli-status` and `make cli-install` only
inside the potpie checkout). Do not use `python -m pip show
potpie-context-engine` for local uv-tool installs. Inspect JSON output when
useful, check API URL/key config, confirm pot scope, and verify source
registration before changing project code.
