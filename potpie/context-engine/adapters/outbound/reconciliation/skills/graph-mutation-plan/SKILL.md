---
name: graph-mutation-plan
description: Cookbook for composing an apply_graph_mutations plan — stable entity_key patterns, the canonical label/edge vocabulary, evidence/invalidation/confidence discipline, and a worked example. Load this when building a non-trivial mutation plan.
version: "1.0.0"
tags: [mutation, graph, ontology, reference]
---

# Composing an `apply_graph_mutations` plan

Call `apply_graph_mutations(plan, event_id, summary)` once per logical group of
mutations. The plan is idempotent on stable `entity_key`s, so a retried plan
converges rather than duplicating. Every structural mutation must stay inside
the given `pot_id` — never reference another pot.

## Plan fields

- `summary`: one-line description of what this plan records.
- `episodes`: `{name, episode_body, source_description, reference_time?}` —
  narrative/source text to retain (e.g. a PR body, a note).
- `entity_upserts`: `{entity_key, labels, properties}`.
- `edge_upserts`: `{edge_type, from_entity_key, to_entity_key, properties}`.
- `edge_deletes`: `{edge_type, from_entity_key, to_entity_key}`.
- `invalidations`: `{reason, target_entity_key?, edge_type?, from_entity_key?,
  to_entity_key?}`.
- `evidence`: `{kind, ref, metadata?}` — what grounds this plan.
- `confidence`: float | null.
- `warnings`: list[str] — anything you could not confirm.

## Stable entity_key cookbook

Use these so re-ingestion (and a later live webhook for a backfilled artifact)
upserts the same node:

| Artifact | Key pattern |
|---|---|
| Repository | `github:repo:<owner>/<repo>` |
| Pull request | `github:pr:<owner>/<repo>:<n>` |
| Issue | `github:issue:<owner>/<repo>:<n>` |
| Module / package | `module:<repo>:<dotted.path>` |
| Feature | `feature:<repo>:<slug>` |
| External ticket / issue | `ticket:<source>:<identifier>` |
| Activity (timeline) | `timeline:activity:<verb>:<short_hex>` |
| Period bucket | `timeline:period:daily:<pot_id>:<YYYY-MM-DD>` |

When no pattern fits, mint a deterministic key from stable identifiers in the
source (never a random id, never a timestamp), so the same fact re-keys the
same way.

## Canonical vocabulary

Always give an entity at least one canonical label — never only generic
`Entity`. Labels/edges outside the canonical vocabulary are downgraded
automatically (entities → `Document` / `Observation`, edges → `RELATED_TO`), so
prefer a canonical type when one fits.

**Entity labels** (topology — source of truth is `domain/ontology.py`):
Repository, Service, Environment, DataStore, Cluster, Team, Person. Plus the
work/knowledge types the playbooks reference: Activity, Feature, Decision, Fix,
BugPattern, Incident, DiagnosticSignal, Module, Document.

**Edge types**: DEFINED_IN (Service→Repository), DEPLOYED_TO
(Service→Environment), DEPENDS_ON (Service→Service), USES (Service→DataStore),
HOSTED_ON (Environment→Cluster), OWNED_BY (Service/Repository→Team/Person),
MEMBER_OF (Person→Team). For completed work, an Activity carries PERFORMED (→
the actor), TOUCHED (→ the modules/features it changed), and IN_PERIOD (→ the
period bucket). Use RELATED_TO only when nothing canonical fits.

## Discipline

- **Justify every mutation** from the event payload or a tool-observed fact. If
  unsure, add a `warning` and keep the plan minimal — don't invent.
- **Supersession → invalidation.** When this event makes a prior fact untrue,
  add an `invalidation` referencing the prior entity/edge rather than silently
  overwriting.
- **Evidence, not vibes.** Link the PR/issue/commit/url that grounds the plan
  under `evidence`; set `confidence` lower when the inference is indirect.

## Worked example — a merged PR that fixes a bug

```json
{
  "summary": "PR #482 fixes the retry-storm in the billing worker",
  "episodes": [{
    "name": "PR #482 body",
    "episode_body": "<the PR description text>",
    "source_description": "github pull_request merged"
  }],
  "entity_upserts": [
    {"entity_key": "timeline:activity:merged:9f3a1c",
     "labels": ["Activity"],
     "properties": {"verb": "merged", "verb_class": "code", "title": "Merge PR #482"}},
    {"entity_key": "fix:billing:retry-storm",
     "labels": ["Fix"], "properties": {"title": "Bound billing retry backoff"}}
  ],
  "edge_upserts": [
    {"edge_type": "PERFORMED", "from_entity_key": "person:github:alice",
     "to_entity_key": "timeline:activity:merged:9f3a1c"},
    {"edge_type": "TOUCHED", "from_entity_key": "timeline:activity:merged:9f3a1c",
     "to_entity_key": "module:o/r:billing.worker"},
    {"edge_type": "RESOLVED", "from_entity_key": "fix:billing:retry-storm",
     "to_entity_key": "bug:billing:retry-storm"}
  ],
  "evidence": [{"kind": "pull_request", "ref": "github:pr:o/r:482"}],
  "confidence": 0.9,
  "warnings": []
}
```

Trivial PRs (typo/lint) need only the Activity — no Fix or Decision. Don't
invent design decisions the PR body doesn't state.
