---
name: graph-mutation-plan
description: Cookbook for composing an apply_graph_mutations plan — stable entity_key patterns, the canonical label/edge vocabulary, evidence/invalidation/confidence discipline, and a worked example. Load this when building a non-trivial mutation plan.
version: "1.1.0"
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

Use the exact key prefixes from the canonical ingestion ontology in your
instructions and reuse identities found by graph reads. PRs and issues are
`Activity` entities with source identifiers, e.g.
`activity:github:pr:<owner>/<repo>:<n>`; implementation paths use `CodeAsset`,
e.g. `code:<repo>:<path>`. Provider references such as `github:pr:...` are
evidence refs, not a substitute for a canonical entity key.

When no pattern fits, mint a deterministic key from stable identifiers in the
source (never a random id, never a timestamp), so the same fact re-keys the
same way.

## Canonical vocabulary

Always give an entity at least one canonical label — never only generic
`Entity`. Labels/edges outside the canonical vocabulary are downgraded
automatically (entities → `Document` / `Observation`, edges → `RELATED_TO`), so
prefer a canonical type when one fits.

The complete public entity/predicate vocabulary is generated from
`potpie_context_core.ontology` and supplied in your instructions. Use its
allowed endpoint pairs and exact prefixes instead of a remembered subset.
Capabilities use `Feature` with supported `PROVIDES`/`IMPLEMENTED_IN` links;
dependencies use topology relations; choices with rationale use `Decision`;
only explicit future-facing prescriptions use `Preference`/`Policy`.

For completed work, `PERFORMED` points from actor to `Activity`, `TOUCHED`
points from activity to the affected scope, and `IN_PERIOD` points from activity
to `Period`. Preserve failed remedies separately from successful fixes. When
the source does not support a specific relationship, defer the candidate instead
of inventing policy or defaulting to a generic association.

## Discipline

- **Justify every mutation** from the event payload or a tool-observed fact. If
  unsure, add a `warning` and keep the plan minimal — don't invent.
- **Supersession → invalidation.** When this event makes a prior fact untrue,
  add an `invalidation` referencing the prior entity/edge rather than silently
  overwriting.
- **Evidence, not vibes.** Link the PR/issue/commit/url that grounds the plan
  under `evidence`; set `confidence` lower when the inference is indirect.

## Worked example — a merged PR that fixes a bug

The example assumes the actor, implementation file, and bug keys were already
resolved by reads.

```json
{
  "summary": "PR #482 fixes the retry-storm in the billing worker",
  "episodes": [{
    "name": "PR #482 body",
    "episode_body": "<the PR description text>",
    "source_description": "github pull_request merged"
  }],
  "entity_upserts": [
    {"entity_key": "activity:github:pr:o/r:482",
     "labels": ["Activity"],
     "properties": {"verb": "merged", "verb_class": "code", "title": "Merge PR #482"}},
    {"entity_key": "fix:billing:retry-storm",
     "labels": ["Fix"], "properties": {"title": "Bound billing retry backoff"}}
  ],
  "edge_upserts": [
    {"edge_type": "PERFORMED", "from_entity_key": "person:github:alice",
     "to_entity_key": "activity:github:pr:o/r:482"},
    {"edge_type": "TOUCHED", "from_entity_key": "activity:github:pr:o/r:482",
     "to_entity_key": "code:o/r:billing/worker.py"},
    {"edge_type": "RESOLVED", "from_entity_key": "fix:billing:retry-storm",
     "to_entity_key": "bug_pattern:billing:retry-storm"}
  ],
  "evidence": [{"kind": "pull_request", "ref": "github:pr:o/r:482"}],
  "confidence": 0.9,
  "warnings": []
}
```

Trivial PRs (typo/lint) need only the Activity — no Fix or Decision. Don't
invent design decisions the PR body doesn't state.
