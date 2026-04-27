# Timeline Subgraph

Phase 8 of the context graph: a first-class temporal layer that captures
*what happened, who did it, to what, and when* across every ingestion source.
The agent consuming the graph via CLI / MCP can slice the project timeline
by actor, subject, branch, verb, or time window — all through the existing
`context_graph` query surface, with no new tools added.

## Motivation

Before this phase, temporality in the graph was indirect: Graphiti edges
carried `valid_at` / `invalid_at`, and the structural layer supported
`as_of` snapshot queries. Agents asking "what's been going on lately?",
"what is Alice working on?", or "what got deployed yesterday?" had no
dedicated tool — they had to pull back event ledger rows and reassemble
the picture.

Timeline fixes that with an explicit subgraph that the ingestion agent
populates on every event.

## Conceptual model

Three node types, three edges:

```
(Actor: Person | Agent | Team)
        │
        │ PERFORMED (source ref, valid_from)
        ▼
   ┌─────────────────────────────────────────────┐
   │ Activity                                    │
   │   verb             (merged_pr, deployed, …) │
   │   occurred_at      (ISO 8601)               │
   │   summary          (one-line narrative)     │
   │   branch?          environment?             │
   │   source_ref       (back to ledger)         │
   │   confidence                                │
   └─────────────────────────────────────────────┘
        │                 │
        │ TOUCHED         │ IN_PERIOD
        │ (multi,         │
        │  wildcard       │
        │  target)        │
        ▼                 ▼
   (subject Entity)   (Period: daily rollup)
```

* **Activity** is the timeline primitive. Every ingestion event that
  represents a project happening emits at least one `Activity` node.
  The verb is a short, stable snake_case string; the summary is one
  line of text written by the plan builder or the deep ingestion agent
  at emission time.
* **Actor** is not a new entity type — it is the existing `Person`,
  `Agent`, or `Team`. The `PERFORMED` edge wires them into a timeline.
  Events without an identifiable actor still produce an Activity; they
  simply lack the `PERFORMED` edge.
* **Period** is a daily rollup bucket. Activities are attached via
  `IN_PERIOD` so pulse-style queries ("the last 7 days") can use a
  compact per-day index. Period summaries default to open + empty, and
  can be filled later (templated or LLM-written) without any schema
  change.

### Why this shape

Every question an agent asks about "recent activity" becomes a
single cheap traversal:

| Question | Cypher shape |
|---|---|
| Global recent pulse | `(:Activity {group_id}) WHERE occurred_at > $since` |
| User-wise timeline | `(:Person {github_login:$u})-[:PERFORMED]->(:Activity)` |
| Feature / subject timeline | `(:Activity)-[:TOUCHED]->(:Feature{name:$f})` |
| Branch / deploy feed | `(:Activity{branch:$b, verb:'deployed'})` |
| Changes to a file | `(:Activity)-[:TOUCHED]->(:CodeAsset{file_path:$p})` |
| "Who merged yesterday?" | `(:Person)-[:PERFORMED]->(:Activity{verb:'merged_pr'}) WHERE occurred_at > $since` |
| Pulse (cached) | `(:Period{period_kind:'daily'}) WHERE opened_at >= $since` |

Activities as the primitive (rather than Period-first) makes every
axis a 1–2 hop query. Periods remain a secondary index: if a Period is
missing or stale, the activity query still works — it just costs a
bit more.

## Verb vocabulary

Verbs are free-form strings (no ontology enforcement) so new sources
can land without a schema change. The built-in plan builders and the
deep agent prefer this vocabulary so queries stay stable:

```
opened_pr       merged_pr        closed_pr        reviewed_pr
authored_commit opened_issue     state_changed    commented
assigned        deployed         decided          declared_progress
declared_completed               performed  (fallback)
```

## Ingestion flow (where Activities come from)

Every Activity is produced by the reconciliation pipeline — not by a
post-ingest hook, not by a separate worker. This keeps the timeline
subgraph as first-class as any other graph fact and means it shares
the same validation, idempotency, provenance, and retry behavior.

Three entry points:

1. **Deterministic plan builders**
   * [`adapters/outbound/reconciliation/github_pr_plan.py`](../../app/src/context-engine/adapters/outbound/reconciliation/github_pr_plan.py)
     emits `merged_pr`, per-commit `authored_commit`, and per-review-thread
     `reviewed_pr` activities.
   * [`adapters/outbound/reconciliation/linear_issue_plan.py`](../../app/src/context-engine/adapters/outbound/reconciliation/linear_issue_plan.py)
     emits `opened_issue`, `state_changed`, `commented`, `assigned`, etc.

2. **Deep reconciliation agent**
   [`adapters/outbound/reconciliation/pydantic_deep_agent.py`](../../app/src/context-engine/adapters/outbound/reconciliation/pydantic_deep_agent.py)
   has ontology instructions for Activity / Period / PERFORMED / TOUCHED /
   IN_PERIOD, so raw CLI episodes and unmapped event sources also produce
   timeline nodes. The LLM writes the one-line `summary` at emission time.

3. **Shared helper**
   [`adapters/outbound/reconciliation/timeline_plan.py`](../../app/src/context-engine/adapters/outbound/reconciliation/timeline_plan.py)
   provides `build_timeline_mutations(...)` — deterministic Activity /
   Period upserts and edges. Plan builders call it; the deep agent can
   emulate the same shape via its structured-output schema.

Activity `entity_key` is deterministic over `(verb, source_ref, suffix)`
so re-ingestion is idempotent and idempotent retries never duplicate
timeline rows.

## Query surface

Timelines are accessed through the same query tool used for every
other family: `context_graph.query(...)` (HTTP `/query/context-graph`,
MCP `context_search` / `context_resolve`).

```python
from domain.graph_query import preset_timeline

# Global pulse for last 7 days
query = preset_timeline(pot_id=p, window="7d", limit=20)

# What is Alice working on this week?
query = preset_timeline(pot_id=p, user="alice", window="7d")

# What touched the auth middleware recently?
query = preset_timeline(pot_id=p, file_path="src/auth/middleware.py",
                        window="30d")

# Recent merges to main
query = preset_timeline(pot_id=p, branch="main",
                        verbs=["merged_pr"], window="7d")
```

All knobs compose. The planner routes `goal=TIMELINE` to the new
`timeline` family (see
[`application/services/graph_query_planner.py`](../../app/src/context-engine/application/services/graph_query_planner.py))
whenever the request isn't anchored to a specific file / function / PR
— for those narrow code-anchored requests, it keeps dispatching to the
existing `change_history` family. Passing `include=["timeline"]`
explicitly forces the timeline path.

### Response shape

```json
{
  "kind": "timeline",
  "goal": "timeline",
  "strategy": "temporal",
  "result": {
    "activities": [
      {
        "entity_key": "timeline:activity:merged_pr:ab12cd34ef567890",
        "verb": "merged_pr",
        "occurred_at": "2026-04-22T10:20:00+00:00",
        "summary": "nandan merged PR #42: Add context graph source resolver policies",
        "branch": "main",
        "source_ref": "source-ref:github:pull_request:potpie/api:42",
        "confidence": 0.95,
        "actors": [{"handle": "nandan", "name": "nandan", "kind": "Person"}],
        "touched": [
          {"entity_key": "github:pr:potpie/api:42", "name": "PR #42",
           "labels": ["PullRequest", "Change"]},
          {"entity_key": "code:file:potpie/api:src/resolvers.py",
           "name": "src/resolvers.py", "labels": ["CodeAsset"]}
        ],
        "period_label": "2026-04-22"
      }
    ],
    "periods": [
      {
        "label": "2026-04-22",
        "opened_at": "2026-04-22T00:00:00+00:00",
        "closed_at": null,
        "lifecycle_state": "open",
        "event_count": 7,
        "top_actors": ["nandan", "ops-dev", "security-lead"],
        "verbs": ["merged_pr", "authored_commit", "reviewed_pr"]
      }
    ],
    "window": {"since": "...", "until": "..."},
    "total_activities": 42
  },
  "meta": {"legs": [{"name": "timeline", "family": "timeline",
                      "strategy": "temporal", "count": 42}]}
}
```

## Schema touch-points

* Ontology: new `Activity` and `Period` entity types, new `PERFORMED`,
  `TOUCHED`, `IN_PERIOD` edge types. See
  [`domain/ontology.py`](../../app/src/context-engine/domain/ontology.py)
  (ontology version `2026-04-phase-8-timeline`).
* Query model: new `preset_timeline()`, new `since` / `until` / `window`
  / `verbs` fields on `ContextGraphQuery`, new `timeline` family on the
  planner. See
  [`domain/graph_query.py`](../../app/src/context-engine/domain/graph_query.py).
* Cypher: `Neo4jStructuralAdapter.get_timeline(...)` in
  [`adapters/outbound/neo4j/structural.py`](../../app/src/context-engine/adapters/outbound/neo4j/structural.py).
* Read helper: `get_timeline(...)` in
  [`adapters/outbound/graphiti/query_helpers.py`](../../app/src/context-engine/adapters/outbound/graphiti/query_helpers.py).
* Executor: `GraphitiContextGraphAdapter._exec_timeline`.

## Benchmarks

Timeline scenarios live in
[`benchmarks/data/scenarios_timeline.json`](../../app/src/context-engine/benchmarks/data/scenarios_timeline.json)
and exercise the user, verb, branch, feature, file-path, empty-window,
and unknown-user slices, plus Period rollup presence. They run as part
of the standard comprehensive benchmark.

## Design notes — what v1 deliberately keeps simple

* **No new MCP tool.** Timeline is a flavour of the same query tool;
  agents discover it via `goal=timeline` / `include=["timeline"]` / the
  timeline preset.
* **Period summaries start empty.** The `summary` slot exists; when a
  period closes, the deep agent (or a scheduled pass) can fill it. The
  query surface does not depend on it.
* **No separate Actor entity.** `Person` / `Agent` / `Team` already exist
  and reconcile across providers — adding a new actor type would fork
  identity. `PERFORMED` wires them into the timeline without changing
  who they are.
* **Activity identity is deterministic.** Re-ingesting the same
  webhook upserts the same Activity. No dedup job required.
* **Ingestion agent owns the summary.** Every ingestion path (including
  the deep agent for CLI episodes) produces the one-line Activity
  summary at emission time — that's when the context is freshest. No
  post-hoc LLM summarization job.

## Extensibility

The schema supports without further change:

* User-declared named periods (`period_kind="declared"`, e.g.
  "auth-rewrite-sprint") via the deep agent.
* New ingestion sources (Slack, deploy webhooks, Sentry) — each adds a
  plan builder or uses the deep agent; no ontology change.
* Causal chains across activities via a future `FOLLOWS` edge between
  `Activity` nodes.
* Per-entity "recent activity" rollups, either denormalized on entity
  properties or materialized as edges, layered on when query patterns
  demand them.
