<!-- potpie-start -->
# Context Engine

This project uses the Potpie context engine for project memory. Read from the graph
before non-trivial work; write durable learnings back after. You are the
intelligence — Potpie validates, lowers, commits, and ranks; it does not infer rich
facts from prose for you.

## Quick Start

```bash
potpie doctor          # check setup
potpie pot pots        # list pots
potpie graph catalog   # discover the graph contract (versions, views, ops)
```

## Graph Surface (CLI) — preferred

Four verbs reach the graph. Pass `--json` for machine-readable output.

```bash
potpie --json graph catalog
potpie --json graph read --view <subgraph.view> [--query "..."] [--scope key:value] [--limit N]
potpie --json graph search-entities "text" [--type Service] [--environment prod]
potpie --json graph mutate --file mutation.json [--dry-run]
```

- **`graph catalog`** — versions, readable views, applicable mutation ops. Start here.
- **`graph read`** — ranked read over a named view (`decisions.preferences_for_scope`,
  `debugging.prior_occurrences`, `recent_changes.timeline`,
  `infra_topology.service_neighborhood`, `decisions.active_decisions`,
  `code_topology.ownership_by_path`, `knowledge.document_context`). Returns
  entities with their relations inline.
- **`graph search-entities`** — resolve an entity's canonical key *before* writing,
  so you link to the existing node instead of duplicating it.
- **`graph mutate`** — validate + apply semantic mutations. Use `--dry-run` first for
  anything not obviously low-risk.

## The Four MCP Tools (legacy wrappers)

When only MCP is configured, use the minimal port — same engine underneath:

- **`context_resolve`** — primary task context wrap (intent/include/mode/source_policy).
- **`context_status`** — cheap pot readiness, freshness gaps, recommended recipe.
- **`context_search`** — narrow follow-up lookup after `context_resolve`.
- **`context_record`** — durable learnings; lowers through the same mutation path as
  `graph mutate`.

Express every read as a `context_resolve` recipe or a `graph read --view`. Do not add
a tool per context type. Valid `include` families: `coding_preferences`,
`infra_topology`, `prior_bugs`, `timeline`, `decisions`, `owners`, `docs`, `raw_graph`.

**Feature:**
```json
{"intent":"feature","include":["coding_preferences","infra_topology","decisions","owners","docs"],"mode":"fast","source_policy":"references_only"}
```

**Debugging:**
```json
{"intent":"debugging","include":["prior_bugs","infra_topology","timeline"],"mode":"fast","source_policy":"references_only"}
```

**Review:**
```json
{"intent":"review","include":["coding_preferences","decisions","timeline","owners"],"mode":"balanced","source_policy":"summary"}
```

## Writing: retrieval-grade descriptions

Every entity and claim you write carries a `description` — a natural-language
**retrieval card** a small local embedder indexes. Write it **for search, not
display**: include the symptoms, synonyms, and scope a future searcher would type.
"wrap external calls in tenacity retry" must say enough that "the payments client
keeps timing out" retrieves it. Resolve identity with `graph search-entities` first,
and never hard-delete a claim — end its validity or retract it.

## Responding To Nudges

A Potpie hook may inject context or a directive (it never reasons — you do):

- **`inject_context`** — treat injected facts as graph truth for this task; don't re-fetch.
- **`instruction`** (e.g. "record the bug+fix if non-obvious") — a *prompt to
  decide*, not an auto-write. Decide truth class, resolve identity, write a
  retrieval-grade description, then call `graph mutate` / `context_record`. If
  nothing durable was learned, do nothing.

## Working Rules

- Begin graph-aware work with `potpie graph catalog` (or `context_status`).
- Start reads `mode=fast` / `source_policy=references_only`; escalate only when
  coverage, freshness, or risk requires it.
- Inspect `coverage`, `freshness`, `quality`, `fallbacks`, `open_conflicts`, and
  `source_refs` before relying on graph memory.
- Keep writes compact and source-reference-first.

## Slash Commands

Use `/potpie-feature` before feature work and `/potpie-record` to capture learnings.
<!-- potpie-end -->
