# Retrieval golden set

This benchmark covers the 15 original retrieval tasks as 19 independent cases:
six demo-store controls, eight Ledgerly resolve tasks, and five identifier
probes from the 2026-09-03 deep-query exercise. Expected answers must stay within
the first five results; identifier probes require the matching resource section
at rank one. Every identifier probe runs even if another probe fails.

Expectations match exact fields: claim subjects and predicates, resource section
keys, and canonical event source references. Nested mentions and description
substrings cannot stand in for the requested answer.

The demo controls isolate the claim-query layer. Each predicate has nine
candidates (three labeled answers and six competing claims), so returning all
candidates without meaningful ranking cannot satisfy top-five recall. Negative
checks also verify that unrelated queries with embeddings disabled fail all six
controls. The Ledgerly cases exercise the public agent-context read path.

The deep-query fixture is replayed through `DefaultGraphService`, the in-memory
graph backend, `ResourceFacade`, and the SQLite resource index. The builder
asserts the source fixture still produces 148 claims (144 live, 4 invalidated)
across 86 entities and 27 predicates. Its ranking clock is fixed at the capture
date so CI results do not decay with wall time.

Run from the repository root:

```bash
uv run pytest benchmarks -k golden
```

Known failures carry their friction item id as strict `xfail`. An unexpected
pass fails CI so the marker must be removed with the fix. Unexpected exceptions
also fail; only assertion failures are accepted as known gaps.

The stronger corpus exposes an additional existing gap: the structured-logging
paraphrase ranks its expected preference eighth with the hashing embedder.
That case is now explicitly marked under R1. The retrieval baseline is six
passes and thirteen expected failures (the original identifier task accounts
for five of them), plus four passing benchmark guard tests. These expected
failures track pending retrieval work; a green job does not mean it is fixed.
