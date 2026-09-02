Record durable Potpie learnings after useful work.

One fix, decision, or preference is one call — no JSON file, no plan:

```bash
potpie record --type fix --summary "<symptom → fix, in the words a searcher would type>" --detail root_cause="<cause>" --detail fix_steps="<step>" --scope service:<name>
potpie record --type decision --summary "<the decision>" --detail rationale="<why>" --scope service:<name>
potpie record --type preference --summary "<prescription>" --detail policy_kind=<kind> --detail prescription="<guidance>" --scope service:<name>
```

`--type` help lists the `--detail` keys each type requires; repeat a key to
build a list. `--scope` takes an existing key — reuse one a read returned, and
run `potpie graph search-entities "<entity name>" --limit 10` (untyped: a wrong
`--type` guess returns nothing) only for a key you have not seen.

For a multi-op batch — topology, timeline events, a fix with its failed
attempts — use the plan flow the workflow skill teaches
(`potpie-infra-architecture`, `potpie-change-timeline`, `potpie-debug-memory`,
`potpie-source-ingestion`):

```bash
potpie graph mutation-template --kind <bug-fix|decision|infra-snapshot|timeline-change|preference-policy>
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
```

Capture: decisions; fixes (with the bug they resolve); bug patterns;
preferences; workflows; incident summaries; source-ingested timeline/doc events.

**Write the `--summary` and `description` for retrieval, not display.** Include
the symptoms, synonyms, and scope a future searcher would type — a vague card
means the fact never resurfaces, and the entity key is minted from the summary.
Pick an honest truth class (`agent_claim` when you inferred it), keep the record
compact and source-reference-first, and never hard-delete — end validity or
retract instead.
