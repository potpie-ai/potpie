---
name: "potpie-graph"
version: "3"
description: "Use when the task can read or write the project-memory graph through the potpie CLI: discover the contract with `graph catalog`, read named views with `graph read`, resolve entity identity with `graph search-entities` before writing, and apply validated semantic mutations with `graph mutate`. Also covers writing retrieval-grade descriptions and responding to nudges. Prefer this over the MCP context_* tools whenever the shell is available."
---

# Potpie Graph Surface (V1.5)

The graph is project memory: preferences, prior bugs and their fixes, infra
topology, decisions, and a timeline of changes. You are the intelligence that reads
it before acting and writes durable learnings after. Potpie validates, lowers,
commits, audits, and ranks. It does **not** scan a repository or infer rich facts
from prose for you.

Always pass `--json` for machine-readable output.

## 1. Discover — `graph catalog`

```bash
potpie --json graph catalog
```

Returns contract + ontology versions, the readable **views**, the **mutation
operations** you can apply (and which are `review_required` or `deferred`), the
entity types, and the predicates. Start graph-aware work here instead of reading
docs. The catalog also reports the active `match_mode` (`vector` vs `lexical`) so you
know whether semantic recall is on.

## 2. Read — `graph read --view`

```bash
potpie --json graph read --view bugs.prior_occurrences --query "refund race timeout" --limit 8
potpie --json graph read --view preferences.active_preferences --scope repo:acme/x,path:src/payments/client.py
potpie --json timeline recent --limit 20
potpie --json graph read --view infra_topology.service_neighborhood --scope service:payments-api --depth 2 --direction out --environment prod
```

Views and what they answer:

| View | Inputs | Answers |
|---|---|---|
| `preferences.active_preferences` | `--scope repo:…,path:…` `--query` | which preferences apply to this code |
| `bugs.prior_occurrences` | `--query` (symptom), optional `--scope service:…` | "seen this before? what fixed it" (bug + fix/PR inline) |
| `recent_changes.timeline` | optional `--scope`, `--since`, `--until`, `--time-window` | recent PRs/tickets/activity for the project pot; use `potpie timeline recent` for the common project-wide path |
| `infra_topology.service_neighborhood` | `--scope service:…` `--depth` `--direction` `--environment` | dependency blast-radius, env-qualified |
| `features.provided` | optional `--scope anchor_entity_key:repo:…` | what a repo/service does (Feature nodes via `PROVIDES` / `IMPLEMENTED_IN`) |
| `decisions.active_decisions` | `--scope` | active decisions |
| `ownership.owner_context` | `--scope` | who owns a scope |
| `docs.reference_context` | `--scope` | reference docs |

Reads return entities **with their immediate relations inline**, so a bug comes back
with its `RESOLVED_BY` fix, and a service with its `DEPENDS_ON` edges — no second
call. Timeline reads default to deduped event rows sorted by occurrence time; use
`--format raw` when you need the underlying relation payloads. Inspect `coverage`,
`freshness`, and `quality` before relying on results.

### Query expansion is your job

The local embedder is small; recall depends on the query. Expand the user's words
before reading: "add retry to the payments client" → also carry "timeout, flaky,
tenacity, backoff, external call". That expansion is in-session reasoning, not
something the daemon does.

## 3. Resolve identity — `graph search-entities`

**Before** linking or asserting against an existing entity, find its canonical key:

```bash
potpie --json graph search-entities "payments api" --type Service --limit 10
```

Reuse the returned `key`. Inventing a near-duplicate key (`service:payments` vs
`service:local:payments-api`) fragments the graph and breaks future reads.

## 4. Write — `graph mutate`

`mutate` applies a batch of **semantic** operations (never raw graph CRUD). Validate
first with `--dry-run` for anything not obviously low-risk.

```bash
potpie graph mutation-template --kind repo-baseline   # schema-only skeleton to fill
potpie --json graph mutate --file mutation.json --dry-run
potpie --json graph mutate --file mutation.json
```

`mutation-template` kinds: `repo-baseline`, `feature`, `preference`, `bug-fix`,
`decision`, `timeline-event` — placeholders only; you fill them from sources you
actually read. Repo/service functionality is first-class: assert
`PROVIDES` (repo/service → `Feature`) and `IMPLEMENTED_IN` (feature → repo/
service/code), each `Feature` carrying a compact `summary` and a retrieval-grade
`description`.

Payload is always batch-shaped:

```json
{
  "graph_contract_version": "v1.5",
  "pot_id": "local/default",
  "idempotency_key": "mutation:bug:settle-deadlock",
  "created_by": {"surface": "cli", "harness": "claude"},
  "operations": [
    {
      "op": "assert_claim",
      "subgraph": "bugs",
      "subject": {"key": "bug_pattern:settle-deadlock", "type": "BugPattern",
                  "properties": {"name": "settle deadlock under concurrent refund"}},
      "predicate": "REPRODUCES",
      "object": {"key": "service:local:payments-api", "type": "Service"},
      "truth": "agent_claim",
      "confidence": 0.9,
      "description": "Concurrent refund + settle on the same order deadlocks the payments DB; shows up as 'refund race timeout' / 'payment deadlock on concurrent settle' under load in prod.",
      "evidence": [{"source_ref": "github:pr:412", "authority": "external_system"}]
    }
  ]
}
```

Applicable ops: `upsert_entity`, `link_entities`, `assert_claim`, `append_event`,
`end_relation_validity`, `retract_claim`.

- `supersede_claim` and `merge_duplicate_entities` always return `review_required` in
  V1.5 (no server-held queue). Re-submit with approval, or drop and instead write a
  low-authority `agent_claim`/observation.
- `patch_entity` and `transition_state` are deferred — model a state change as a new
  claim or `append_event` (e.g. a `VERIFIED` event), never an in-place edit.
- Never hard-delete a claim: use `end_relation_validity` or `retract_claim`.

### Truth classes

Pick the truth class honestly — it feeds the ranker:

`authoritative_fact` (explicit source of truth) · `source_observation` (observed
source data read by the harness) · `user_decision` (a person decided) ·
`preference` · `agent_claim` (you inferred it; default when unsure) ·
`timeline_event` (something happened) · `quality_finding`. Durable writes need
evidence **or** an explicitly low-authority truth class.

Do not use the graph as a deterministic code scanner. If a repo, PR, ticket, log,
or document should become memory, the harness reads that source, decides what is
worth recording, resolves identity, and writes a semantic mutation.

### Retrieval-grade descriptions (the one rule that matters most)

Every entity and claim carries a `description` — a natural-language **retrieval
card** the local embedder indexes. Write it **for search, not display**: include the
**symptoms, synonyms, and scope** a future searcher would type. Validation only
*warns* on a weak description, but a vague card means the fact never resurfaces.
Compare:

- Weak: `"deadlock fix"`
- Strong: `"Concurrent refund + settle deadlocks payments DB under load; seen as 'refund race timeout' and 'payment deadlock on concurrent settle' in prod; fixed by ordering lock acquisition in services/payments/settle.py"`

## Responding To Nudges

A Potpie hook may call `graph nudge` and inject its result into your session. The
hook never reasons — you do.

- **`inject_context`** → treat the injected facts as graph truth for this task; they
  were ranked for your current scope, so use them rather than re-fetching.
- **`instruction`** (e.g. "you resolved `<error>` after editing `<files>` — record
  the bug+fix if non-obvious", or "capture durable learnings") → a *prompt to
  decide*, not an auto-write. Decide the truth class, resolve identity with
  `graph search-entities`, write a retrieval-grade `description`, then `graph mutate`.
  If nothing durable was learned, do nothing.

Writes are idempotent by `idempotency_key`, so a nudge-driven capture you've already
made will not duplicate.
