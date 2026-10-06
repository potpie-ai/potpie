Record durable Potpie learnings after useful work.

A fix or a free-form note is one call — no JSON file, no plan:

```bash
potpie record --type fix --summary "<symptom → fix, in the words a searcher would type>" --scope service:<name>
potpie record --type workflow --summary "<the workflow, in the words a searcher would type>" --scope service:<name>
```

`record` takes `--type`, `--summary` and `--scope` only, so it refuses a type
that needs structured fields — `decision`, `preference`, `policy`,
`bug_pattern`, `verification`. `--scope` takes an existing key — reuse one a
read returned, and run `potpie graph search-entities "<entity name>" --limit 10`
(untyped: a wrong `--type` guess returns nothing) only for a key you have not
seen.

For a decision, a preference, or a multi-op batch — topology, timeline events, a
fix with its failed attempts — use the plan flow the workflow skill teaches
(`potpie-project-preferences`, `potpie-infra-architecture`,
`potpie-change-timeline`, `potpie-debug-memory`, `potpie-source-ingestion`):

```bash
potpie graph mutation-template --kind <bug-fix|decision|infra-snapshot|timeline-change|preference-policy>
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
```

When `propose` answers `review_required`, ask the user and add
`--approved-by <user-ref>` to the commit.

Capture: decisions; fixes (with the bug they resolve); bug patterns;
preferences; workflows; incident summaries; source-ingested timeline/doc events.

**Write the `--summary` and `description` for retrieval, not display.** Include
the symptoms, synonyms, and scope a future searcher would type — a vague card
means the fact never resurfaces, and a fix's entity key is minted from its
summary. Pick an honest truth class (`agent_claim` when you inferred it), keep
the record compact and source-reference-first, and never hard-delete — end
validity or retract instead.
