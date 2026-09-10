# CLI fixes and retrieval experiments — 2026-09-10

**The first fixes improve the agent's normal path: documents includes source text, strong lexical matches survive ranking, parallel environment claims survive writes and neighborhood reads, and passage responses contain usable evidence.** The remaining trust gap is distinguishing unsupported questions from weak candidates and reliably attaching mandatory preferences to a task.

These changes are implemented in the working tree and built into isolated wheels. They are not merged or installed over the user's normal CLI. The baseline was already dirty; existing work, including definition retrieval and resource filtering, was preserved.

## Measured changes

Replayed the same 45 commands against a fresh synthetic pot, `local:cli-usage-fixed` (`pot_75f168da217d`), with four documents and 12 chunks. Added seven fresh query probes and a separate two-claim environment pot. Rank below is the first answer-bearing item in ordered JSON; it can be a section summary or its source chunk. Full evidence and assertions are in [results.json](results.json).

| Case | Baseline | Final build |
|---|---|---|
| Exact body token `ORCHID-731`, default search | Rank 23 | Rank 1 |
| Known rollback phrase, default search | Rank 22 | Rank 1 |
| Natural rollback task | Rank 4 | Rank 1 |
| PMS definition with `--include docs` | Correct document absent | Correct passage first |
| Infrastructure commit readback | 35/37 claims | 37/37 claims |
| Minimal prod/staging reproduction | One claim retained | Both verified and visible in neighborhood |
| Unknown include / invalid read direction | Exit 0, empty candidates | Exit 1, actionable validation error |
| Text `document_passages` | `[?]`, identifiers, no excerpt | `ResourceChunk`, excerpt, source reference, fetch command |

The final query-matrix median was 247 ms, compared with 246 ms in the baseline. This is a local observation, not a performance benchmark. New pot IDs, write times, and the baseline's extra record mean the pots are not byte-identical snapshots.

## What the experiments changed

### Preserve claim identity through the entire path

The baseline report attributed the environment loss to storage identity. Investigation found an earlier cause: entity canonicalization deduplicated edges solely by predicate and endpoints, dropping one environment **before storage**. It now preserves claim key, environment and source identity.

A second storage defect affected parallel semantic claims sharing a source: relationship `MERGE` did not include `claim_key`. The writer now includes it for canonical claims and limits vector updates to the same claim. Legacy writes without a claim key retain their previous matching behavior. Reasserting existing keyed claims remains idempotent.

Finally, the live read exposed a third collapse: `graph neighborhood` deduplicated edges by endpoints and predicate. FalkorDB inspection now retains claim identity (or a legacy edge UUID); the in-memory projection also retains canonical claim identity. A depth-two walk returns each claim once, with both environments visible.

Real FalkorDB tests cover same/different sources, same/separate batches, reassertion, independent embeddings, and neighborhood readback. The minimal pot now verifies `ok: true`, reads back two claims, and shows prod and staging in the final neighborhood. The larger fixture still has `verification.ok: false`, `status: degraded` because of one duplicate-candidate quality finding; **its missing-claim list is empty**. Verification outcome normalization remains separate work.

This prevents new losses. Previously discarded claims cannot be recovered from the graph alone; they need to be reasserted from source evidence with a new reviewed mutation. No historical user data was modified.

### Make document lookup match the agent's expectation

At the agent-facing service boundary, explicit `--include docs` expands to `docs,resources`. `--include resources` remains text-only. Named graph views keep their precise summary/passage meanings. Include normalization handles case and avoids duplication. CLI help and skill recipes describe this behavior.

This is deliberately a user-facing contract change: callers requesting `docs` can now receive resource items as well as graph claims. Consumers must use each item's `include` and payload shape. Named `knowledge.document_context` remains available for summaries alone.

### Protect strong matches without replacing definition ranking

Compared two candidates on saved baseline responses:

1. Removing family weights promoted 7–17 document/resource candidates in the rollback cases, and seven candidates for an unrelated negative query.
2. Giving a score floor only to resource candidates with a lexical hit and at least 80% query-term coverage promoted the relevant passage in each target case and none in that negative query.

The implementation uses the second candidate for ordinary retrieval. Explicit/inferred docs intent also gives documents and resources equal family weight. Responses expose `reader_score`, `include_weight`, and `lexical_coverage_floor` so the change is inspectable. [Offline comparison](ranking-comparison.json).

Broader definition tests caught an unsafe interaction: a negated acronym mention could outrank an authoritative definition after lexical promotion. The final implementation disables this floor for definition intent and preserves the existing definition-specific ranking. All definition regressions pass.

This is a retrieval heuristic, not a calibrated relevance or answer probability. A term match can still be a distractor, and docs-intent weighting changes negative-query ordering even when nothing receives lexical promotion. It needs evaluation on a larger corpus before claiming general ranking quality.

### Return evidence an agent can act on

Passage projection now preserves a normalized resource type, snippet, source reference, chunk IDs, retrieval metadata and fetch command. Compact output retains these fields, and text prints the excerpt and fetch command. See the [actual text response](passage-text.txt).

The fetch command is still unscoped; agents working on an explicitly selected pot should carry forward `--pot local:<name>`. A future response contract should include the effective pot in every suggested read command.

Invalid include names and invalid named-read directions now fail instead of masquerading as a successful empty lookup. Wrong-case entity types are still unresolved.

## Skills and agent expectations

Updated the installed agent-bundle skills and corresponding Claude copies: graph version 8, CLI version 7, and markdown/PDF/spreadsheet resource versions 4. Corrections cover:

- Documents search summaries and indexed source text; a missing summary does not make indexed text unsearchable.
- Passage reads have relative filtering and semantic-threshold controls; they do not always fill the requested limit.
- `Adapter` exists; decision scope follows the actual anchor.
- Status reports effective and active pots; skill readiness uses `status --harness codex`, while setup/installation use `--agent`.
- `doctor` does not accept `--pot`; malformed include/direction inputs are errors.

Rebuilt wheels and installed the skills into an isolated Codex harness. [Installed skill snapshots](installed-skills/) preserve what the agent receives. Potpie's catalog loads their versions successfully. The generic skill-creator frontmatter validator rejects Potpie's existing custom `version` field; that validator did **not** pass. We retained the field because Potpie's skill catalog consumes it.

The main lesson is that version equality only checks whether installed text matches packaged text. It says nothing about whether that text teaches correct behavior. Keep the normal path short: resolve a task, search a phrase, fetch the source, record a learning. Validate those recipes by asserting expected evidence in responses, not just successful exit codes. Consolidating duplicated skill contracts into a single source remains follow-up work.

## Additional probes and remaining gaps

Seven new queries checked near-miss identifiers, acronym specificity, document-filter casing and natural paraphrases. [Full responses](holdouts.jsonl.gz).

- `ORCHID-732` did not receive lexical promotion. It still returned unrelated candidates.
- CRM found its own glossary first. Unknown XYZ returned weak unrelated passages; absence handling remains incomplete.
- Rollback-command, monthly-cost and rollback-hold paraphrases each placed the appropriate source first.
- Uppercase `--include DOCS` found the PMS passage.
- Negative-query weak-match warnings remain present. Rows and high coverage must not be treated as proof of an answer.

The next experiments should target these independently:

1. **Mandatory preferences:** retrieve applicable policy by scope independently of task similarity, then attach it to resolve. Keep policy applicability distinct from ranked task evidence. Current scope-only workaround works; default resolve still misses the fixture preference.
2. **Unsupported answers:** expose `weak_evidence`/`no_candidates` at the top level, retain candidate evidence, and calibrate any abstention rule against exact-token, paraphrase, negative and negated-definition cases. A single cosine threshold already fails the exact-token case.
3. **Outcome contract:** separately report write applied, readback complete, and quality findings, with documented exit semantics. Do not equate a degraded quality report with a missing write.
4. **Discoverability:** document inventory, entity-type validation, harness persistence and architecture-question routing still need work.

This was a same-agent diagnostic evaluation on synthetic data, not an independent agent-success study. No managed backend, live Neo4j server, clean dependency installation, PDF extraction or changed-content replacement was tested.

## Verification and reproduction

**470 unique focused tests passed:** 236 CLI, 170 engine, 44 core, 12 real FalkorDB integration and 8 inspection unit tests. Logs are included. The integration log also includes seven of the inspection tests; these are not double-counted. Ruff passed on the changed scope using the existing assertion/observability exceptions, and `git diff --check` passed. Final wheel hashes are in [results.json](results.json).

Review addressed correctness (identity preservation, definition negation, compact evidence), compatibility (keyed versus legacy writes, expanded docs results), architecture (document expansion at the agent boundary, resource projection in a small module), security (bound Cypher parameters, closed-vocabulary validation) and performance (no extra network calls for ranking; bounded existing traversal). No dependencies were added. Ranking generalization and the explicitly listed contract gaps remain limitations.

Use the experiment build from the repository root:

```sh
./.tmp/cli-usage-experiments-20260910/potpie search ORCHID-731 --pot local:cli-usage-fixed
./.tmp/cli-usage-experiments-20260910/potpie resolve "What does PMS stand for?" --include docs --pot local:cli-usage-fixed
./.tmp/cli-usage-experiments-20260910/potpie graph neighborhood --entity service:probe-api --detail full --pot local:cli-env-fixed
```

Replay the matrix against that build, or add `--seed` with a new pot name to create another fresh corpus:

```sh
python3 docs/evaluations/cli-usage-2026-09-10/replay.py \
  --cli ./.tmp/cli-usage-experiments-20260910/potpie \
  --pot local:cli-usage-fixed --output /tmp/potpie-fixed-replay.jsonl
```

The [final 45-query transcript](queries-final.jsonl.gz) comes from the reviewed build. [Seed and first candidate](seed-and-first-candidate.jsonl.gz) records initialization before the definition guard and neighborhood correction. [First follow-ups](followups-first-candidate.jsonl.gz) deliberately preserve the one-row neighborhood response that revealed the projection defect; [final neighborhood](minimal-read-final.json) shows its correction. The fresh query probes ran after the definition guard and before the inspection-only correction.
