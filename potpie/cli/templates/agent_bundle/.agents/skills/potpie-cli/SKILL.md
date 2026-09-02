---
name: potpie-cli
version: "6"
description: "Use when the task is centered on running, explaining, configuring, or troubleshooting the `potpie` command: doctor, login, pot management, source registration, search, graph workbench reads/writes, resource (document payload) commands, and pot scope behavior."
---

# Potpie CLI

Use this skill when the user is asking about the `potpie` command itself. For
ordinary project-memory context, prefer the relevant use-case skill first.

## Setup And Scope

Install the **published** package with `uv tool install 'potpie[all]'` or
`pip install 'potpie[all]'`. Only inside the potpie source checkout — the repo
whose `Makefile` has a `cli-install` target — reinstall from source with:

```bash
make cli-install    # UI build + stop old daemon + editable install
make cli-status
potpie doctor
```

Anywhere else there is no Makefile and `make cli-status` is a dead call; the
install facts come from `uv tool list` and `which -a potpie`. Do not use raw
`uv tool install --editable …` or `pip install` for day-to-day reinstalls of
the checkout.

The extras are not optional decoration. Bare `potpie` installs a **remote-only
client**: the CLI and the RPC transport, with no graph-native local backend and
no daemon. On such an install the local host runs in process on the
dependency-free `embedded` backend, `potpie daemon …` reports that
`potpie[daemon]` is missing, and the first run is
`potpie setup --remote <url> [--token <key>]` rather than `potpie setup`. If
`potpie backend doctor` names a missing driver, that is a packaging answer —
report the extra it names; do not treat it as a broken graph.

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
in a few lines; `doctor` adds the install, the repo → pot mapping and the
resource store, and exits 0 whenever it produced a report.

Pot scope for `graph …`, `doctor`, `resolve`, `search`, `record` and every read
resolves in this order:

1. Explicit `--pot`.
2. Repo-local default set by `source add repo` or `pot default set`.
3. Registered repo source matching the current working tree path or
   `remote.origin.url`.
4. Active pot from `potpie pot use`.
5. Clear failure asking for setup, source registration, default selection, or
   explicit `--pot`.

`potpie status` reports the *active* pot (step 4) rather than the repo default,
so when the two differ, trust the header of a read or `doctor` for the pot a
read will hit.

A `--pot` ref may carry its origin: `local:<name-or-id>` or `managed:<name-or-id>`.
Pass it that way once a read header has named the pot — a bare name is checked
on both origins whenever a managed host is configured, which costs a round trip
per command, and a name that exists on both is refused as ambiguous.

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

## Context Verbs

```bash
potpie status
potpie resolve "<task>"
potpie resolve "<task>" --intent debugging --include prior_bugs,docs,timeline
potpie search "query"
potpie --json search "query"
potpie search "query" --include decisions,features
potpie record --type <fix|decision|preference|bug_pattern|verification> --summary "<…>" --detail <key>=<value> --scope service:<name>
```

`resolve` is the first read for a task: the intent is inferred from the task
text when `--intent` is omitted, and the reply is a bounded envelope of
`subject PREDICATE object · fact` rows across families with a `+N more` footer.
`search` is the follow-up for a known phrase; its query is positional — there
is no `--query`. Bare search resolves intent `unknown`, which ranks the phrase
across all nine families with the same ranker as `resolve`, so a phrase from a
document can sit below recent timeline rows: `--include docs` (or
`--intent docs`) narrows it, and `--include` names any family directly.
`--help` lists both vocabularies. `confidence` in either header is coverage,
not a verdict; a small pot reads `low` with the right rows on top. `record` is the one-call write for a fix, decision,
preference, bug pattern or verification; `--type` help names the `--detail`
keys each type requires, and a repeated `--detail` key builds a list.

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
with no graph query and batches multiple ids in one call. `list` requires
`--doc`. `rm` is destructive and requires `--confirm`. `potpie doctor` reports
resource store readiness. Ingestion flows live in the per-format
`potpie-resource-*` skills.

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
`resource import`, and anything a script parses. `catalog --task` is accepted
and ignored, and the text catalog (under 1 KB) already lists views and
mutation ops. `describe --examples` renders only with `--json` and carries
read examples only; the write payload shape is `mutation-template`.
`commit --verify` prints the plan id, readback and quality status, so
`graph history --plan <plan_id>` is for later inspection.

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
written with `potpie record` or graph workbench mutations. Do not use pot-level
connector queueing or deterministic local code scans as the agent ingestion path.
Do not use scanner-driven graph updates.

For CLI failures, stay in this skill: run `potpie status`, `potpie doctor`, or
`uv tool list` for install facts (`make cli-status` and `make cli-install` only
inside the potpie checkout). Do not use `python -m pip show
potpie-context-engine` for local uv-tool installs. Inspect JSON output when
useful, check API URL/key config, confirm pot scope, and verify source
registration before changing project code.
