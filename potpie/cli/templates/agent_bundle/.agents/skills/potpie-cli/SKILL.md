---
name: potpie-cli
version: "4"
description: "Use when the task is centered on running, explaining, configuring, or troubleshooting the `potpie` command: doctor, login, pot management, source registration, resolve/search/record, graph workbench reads/writes, resource (document payload) commands, and pot scope behavior."
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
check: daemon, pot, backend readiness and claim counts in a few lines; `doctor`
adds the install, the backend's capabilities, the repo → pot mapping, and the
resource store and its index (`resources` and `resource_index` rows).

Pot scope for `graph …`, `resolve`, `search`, `record` and the `resource`
commands resolves in this order:

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

`source add` accepts a closed set of kinds: `repo`, `linear`, `jira`,
`confluence`, `notion`, `url`. A repo on any host is `repo` — `github`,
`gitlab`, `gitbucket` and `git` canonicalize to it, and the reply names the
kind you typed as `requested_kind`. `--default` / `--no-default` are repo-only;
an explicit `--default` on another kind fails with `repo_default_not_applicable`.
An unknown kind exits 1 with `unknown_source_kind`. A document is not a source:
`potpie source add pdf ./q3.pdf` exits 1 with `source_kind_is_a_document`. Use
the matching `potpie-resource-*` skill and `potpie resource import` instead.

## Context Verbs

```bash
potpie status
potpie resolve "<task>"
potpie resolve "<task>" --intent debugging --include prior_bugs,docs,timeline
potpie search "query" --include docs
potpie --json search "query" --include decisions,features
potpie record --type fix --summary "<symptom → fix>" --scope service:<name>
```

Use one shared discovery pass across skills: run `resolve` concurrently with
scope-only `preferences_for_scope --repo current` for code work and untyped
`graph search-entities` for explicitly named entities whose keys are unknown.
Reuse current results and hook context for the same task, pot, and scope. Run
`status` alongside discovery if health needs checking; resolve ambiguous pot
routing first and check pot IDs before combining results. Use returned keys for
focused views and batch returned chunk IDs into `resource get`; those
follow-ups can run concurrently once inputs are known. Stop when evidence and
constraints are covered. Load the relevant use-case skills together; do not
repeat resolve for each skill.

`resolve` is the broad discovery read: it reads the families of `--intent`
(default `feature`; the intent is not inferred from the task text, so pass
`debugging` for a failure or `operations` for what changed) and returns a
bounded envelope of `[family] fact` rows. `search` is the follow-up for a known
phrase; its query is positional — there is no `--query`. Bare, it reads
infra, timeline, decisions, docs and resources; name the family you want with
`--include` to narrow it. `--include docs` searches both section summaries and
document text; `--include resources` searches text only. An unknown include
name comes back as `unknown_include`. `confidence` in either header is
not a verdict; a small pot reads `low` with the right rows on top. `record` is
the one-call write for a fix or a free-form note (`workflow`, `service_note`,
`runbook_note`, …): it takes `--type`, `--summary` and `--scope` only, so a
decision, preference, bug pattern or verification goes through
`graph mutation-template` and a plan.

## Resources (document payloads)

```bash
potpie --json resource import <dir> --doc <slug> --source-ref <uri> --source-kind pdf
potpie resource get potpie://res/<doc>/<section>/0000 --with-neighbors
potpie resource list --doc <slug>
potpie resource rm <slug> --confirm
potpie resource index status
potpie resource index build --wait
```

`import` absorbs a chunk directory an extraction script produced — atomic,
re-import replaces and bumps `revision`, and the document's structure is
written to the graph in the same command. `get` resolves chunk ids
(`potpie://res/<doc>/<section>/<seq>`, optionally `@rev<N>`) to text with no
graph query, batching up to 128 ids in one call; its output is bounded and
`--full` lifts the budget. `list` requires `--doc`. `rm` is destructive: without
`--confirm`, a `--json` or non-interactive run fails with
`destructive_confirmation_required`. `pot reset` and `pot archive` clear a pot's
stored documents along with its graph state. `index status` reports the index
profile and pending embeddings; `index build --wait` embeds now, and
`index rebuild --confirm` re-derives the index from the stored files.
`potpie config set resource_index <profile>` picks `sqlite_hybrid`,
`sqlite_fts` or `none` (`CONTEXT_ENGINE_RESOURCE_INDEX` overrides it).
Ingestion flows live in the per-format `potpie-resource-*` skills.

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

One rule for `--json`: text for reads, `--json` for `propose`, `commit`,
`resource import`, and anything a script parses. `catalog --task` does not
narrow the catalog. Before ingestion, `graph catalog --profile full` exposes all
public entity types with their identity policy, predicates, and allowed
endpoints; `--profile read` stays compact for read-view discovery. Use the full ontology and the selection
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
