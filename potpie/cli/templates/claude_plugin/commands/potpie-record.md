Record durable Potpie learnings after useful work.

Choose the workflow skill first: `potpie-project-preferences`,
`potpie-infra-architecture`, `potpie-change-timeline`, `potpie-debug-memory`, or
`potpie-source-ingestion`.

First resolve identity so you link to existing nodes, not duplicates:

```bash
potpie graph search-entities "<entity name>" --type <Service|BugPattern|...>
```

Then write through the V2 graph plan workflow:

```bash
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

Capture:

- decisions
- fixes (with the bug they resolve)
- bug patterns
- preferences
- workflows
- incident summaries
- source-ingested timeline/doc/reference events

**Write the `description` for retrieval, not display.** Include the symptoms,
synonyms, and scope a future searcher would type — a vague card means the fact never
resurfaces. Pick an honest truth class (`agent_claim` when you inferred it), keep the
record compact and source-reference-first, and never hard-delete — end validity or
retract instead.
