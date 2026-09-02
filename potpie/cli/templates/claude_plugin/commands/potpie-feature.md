Use Potpie context before feature work. Load `potpie-project-preferences` first,
and load `potpie-infra-architecture` when the change touches services,
deployment, adapters, or runtime behavior.

If the `potpie` CLI is available, two reads cover it:

```bash
potpie resolve "<the feature in the user's words>"
potpie graph read --subgraph decisions --view preferences_for_scope --repo current --limit 12
```

`resolve` returns decisions, features, infra and prior bugs as
`subject PREDICATE object` triples in one bounded envelope; preferences are read
by scope with no `--query`, because that view applies an absolute similarity
floor a task sentence rarely clears (`resolve` applies it too, so the second
read is not optional). Decisions anchor on services, so for the full list
read `--view active_decisions --scope service:<name>`, not the repo scope.
Inspect coverage, quality, and source refs before relying on the result, and
pass `--pot local:<name>` once the first read has named the pot.
