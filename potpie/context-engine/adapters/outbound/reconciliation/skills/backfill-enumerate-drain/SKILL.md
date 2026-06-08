---
name: backfill-enumerate-drain
description: How to backfill a newly-added repo or team — the enumerate-then-drain loop, the single-event contract, and resume-safe todo discipline. Load this when handling a *.added seed event with the planner on.
version: "1.0.0"
tags: [backfill, planner, todo, ingestion]
---

# Backfill: enumerate, then drain

You are handling a single `*.added` **seed** event (a repository or a team was
just connected). The graph has no history for it yet. One seed fans out into
many artifacts (PRs, issues, projects, documents). Your todo/plan tools are ON.
Use this procedure for any such backfill; the playbook tells you WHICH
enumerators and WHAT to extract per artifact.

## The loop

1. **Enumerate (one call per list tool).** Call each list/enumerator tool the
   playbook names exactly once. They are bounded server-side to a trailing
   window and a hard item cap, and come back newest-first. Do **not** page or
   scrape beyond what one call returns — the window/cap is deliberate; the tail
   arrives via live webhooks and future backfill.
2. **Write one todo per returned ref**, across all enumerated kinds. If the
   playbook says some kinds frame the others (e.g. projects/documents frame
   issues), drain those first.
3. **Drain newest-first.** For each todo: hydrate the ref with its detail tool
   (`github_get_pull_request` / `github_get_issue` / `linear_get_*`), apply the
   extraction the relevant per-kind playbook prescribes via
   `apply_graph_mutations`, then mark the todo done and move on.

## Resume discipline (critical)

The todo list rides in your message history, which is checkpointed after every
tool call. A resumed run therefore **continues** an existing list — if todos
already exist, never re-enumerate and never restart the walk from scratch.
Re-running an enumerator on resume duplicates work and burns budget.

## Single-event contract

This batch contains exactly **one** event — the seed. Per-artifact identity
lives in each artifact's stable `entity_key`, **not** in event ids. So:

- Pass the **seed's** `event_id` to every `apply_graph_mutations` call.
- Finish with a single `mark_event_processed(seed_event_id, summary)` then
  `finish_batch`.

Stable keys make a backfilled artifact and a later live webhook for the same
thing converge on one node instead of duplicating. See the
`graph-mutation-plan` skill for the key patterns.

## Budget discipline

Breadth of a coherent recent seed beats exhaustive depth. If you approach the
tool budget, ingest what you already have cleanly, then `mark_event_processed`
the seed and `finish_batch` — the remainder re-batches later. Never fabricate a
PR/issue/project/document/comment you did not actually fetch; add a warning
instead.
