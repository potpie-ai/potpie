# Potpie CLI and agent usability evaluation — 2026-09-10

**Potpie can store, find, and fetch useful context, but its default query surface and shipped skills still require too much insider knowledge.** Targeted retrieval worked; broad search often buried the answer. A separate, reproducible write defect loses one of two environment-qualified claims. I would fix that before trusting multi-environment infrastructure memory.

This is an evaluation of the existing dirty working tree, not a claim that the product defects below were fixed. No production implementation or pre-existing edits were changed during this evaluation.

**Follow-up:** [Implemented fixes and experiments](experiments/README.md) include rebuilt wheels, fresh seeded-pot results and regression tests. Investigation corrected the write-loss diagnosis below: endpoint-only deduplication first discarded claims during canonicalization; same-source storage identity and neighborhood projection also needed fixes. This baseline report preserves the original observations.

## What was actually built and tested

- Built fresh wheels for `potpie 2.0.0`, `potpie-context-core 0.1.0`, and `potpie-context-engine 0.1.0` from revision `4936b6cee2578641698b8167ef2dc7498dc10064` plus the existing uncommitted changes. Wheel hashes are in [results.json](results.json).
- Installed those wheels into an isolated import target, using the existing repository Python 3.12.13 dependency environment. This tests the built packages; it is **not** a clean dependency-install test.
- Started a separate local daemon and state directory. Backend: FalkorDB Lite. Resource index: SQLite hybrid. Embedder: `all-MiniLM-L6-v2`, 384 dimensions, using an existing local model cache.
- Created `local:cli-usage-eval` (`pot_b10a82caa287`): **126 claims, 79 entities, 25 predicates; 4 documents, 12 sections, 12 chunks** after the record test. Sources are synthetic fixtures, not production facts.
- Reused the Ledgerly synthetic baseline, infrastructure, decisions, preferences, bugs, and timeline corpus from `.tmp/improvements/deep-query`; imported its runbook, ADR, and CSV cost report. Added a fresh Unicode acronym document, misleading glossary material, and an exact token present only in unsummarized chunk text.
- Installed all 11 packaged Codex skills into a separate harness home; none showed version drift. Created an empty control pot for isolation checks and a third, two-operation pot for the environment collision reproduction.
- Captured **109 CLI invocations**, including a **45-command query matrix**, text/JSON comparisons, errors, imports, commits, record/readback, and daemon restart. Warm query-matrix median was **246 ms**, p95 **269 ms**; the first semantic read after restart took **2.427 s**. These are local observations, not a general performance benchmark.
- Ran **156 focused CLI contract tests**, all passing in 4.24 s. [Test log](contract-tests.txt).

The isolated executable remains at `.tmp/cli-usage-eval-20260910/potpie`. The normal global CLI and existing pots were not reconfigured. Every command below uses `potpie` as shorthand for that wrapper; the compressed transcript preserves the exact executable paths and arguments.

## Retrieval journeys

Rank means position after the text presenter's claim deduplication. The default text response shows only the first ten rows. “Found” means the source needed for the question surfaced, not merely that the command returned data.

| Agent's task | Default experience | Targeted follow-up |
|---|---|---|
| Why are customers charged twice after Redis failover? | Useful incident first; fix at rank 4; failed attempt also returned. | `debugging.prior_occurrences` returned the bug/fix context. |
| How do I roll back a bad ledger-api deploy? | Correct rollback section/chunk at ranks 4–5, below three advisory-lock decision rows. | `resolve … --include docs` put the rollback section first. |
| Why advisory locks instead of a single-writer queue? | “Why” inferred debugging; the relevant ADR resource was absent. A timeline event mentioned the decision without answering the tradeoff. | `resolve … --include docs` put the ADR context first, including the rejected queue's 300 ms overhead. |
| Search the known phrase “disable Argo auto-sync after rollback” | Correct section/chunk at ranks 22–23: invisible in default text. | `search … --include docs` returned the right section first. |
| What does PMS stand for in this project? | Definition intent found the correct chunk first: Prüfung/Montage/Service. | Adding `--include docs` lost that document entirely. `--include resources` or `docs,resources` restored it. |
| Find exact body-only token ORCHID-731 | Broad search placed the matching resource at rank 23, behind unrelated graph memory. | `search ORCHID-731 --include resources` and `document_passages` each returned one matching chunk. |
| Who owns notifier? | Correct Growth ownership first. JSON duplicated the same claim across families; text deduped it. | Predicate-filtered neighborhood returned the ownership directly. |
| Add a posting path safely; apply project preferences | Default resolve omitted the seeded mandatory testing preference. | Scope-only `preferences_for_scope` returned it; adding the natural task as `--query` removed it again. |
| What changed in the last 30 days? | Returned relevance-ranked context, not a complete ordered timeline. | Named timeline view returned the five fixture activities within the window in date order. |
| What is Plaid's monthly cost? | Tested with `--include docs`: correct cost section first, $2,400/month. | Stored CSV remained available by chunk ID. |

This is a small, deliberately diagnostic corpus. It does not establish a production recall percentage or the success rate of independent agents. The same agent inspected the code and ran the workflows; this was not a blinded multi-agent study.

## Findings, in priority order

### 1. Critical: environment-qualified facts collide during a write

The larger infrastructure plan proposed 37 claims; commit verification found only 35. Both missing claims were production configurations with staging counterparts for the same service/config pair. Full reads retained the staging claims and omitted production.

I reproduced this independently with [two operations](fixtures/minimal-environment-collision.json): the same service `CONFIGURES` the same config variable in `prod` and `staging`, with different facts and source references. The proposal had two distinct claim keys. Commit returned `status: committed`; verification failed; neighborhood contained **one** relation, `BANK_ENV=sandbox in staging`.

```sh
potpie --host local pot create cli-usage-env-probe
potpie --json graph propose --file docs/evaluations/cli-usage-2026-09-10/fixtures/minimal-environment-collision.json --pot local:cli-usage-env-probe
potpie --json graph commit mutation-plan:110b085d1fea --verify --pot local:cli-usage-env-probe
potpie --json graph neighborhood --entity service:probe-api --detail full --pot local:cli-usage-env-probe
```

Use the newly returned plan ID when reproducing; stored plans expire.

The observed behavior is consistent with [the writer's MERGE identity](../../../potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/cypher.py): it uses group, predicate, endpoints, and storage `source_ref`, but neither environment nor canonical claim key. Subsequent properties overwrite the same relation. This is a write identity problem, not a query-ranking problem. Preserve canonical claim identity through persistence and add a real-backend regression covering both environments in one batch and across batches.

Evidence: `commit-02-infra`, `infra-raw`, `config-canonical`, `collision-propose`, `collision-commit`, `collision-read` in the transcript.

### 2. High: “docs” excludes the document text an agent expects to search

`docs` means graph section summaries. `resources` means indexed chunk text. This distinction is technically coherent but surprising at the user boundary, especially because the skills repeatedly recommend `--include docs` as the way to search documents.

```sh
potpie resolve "What does PMS stand for in this project?" --pot local:cli-usage-eval
potpie resolve "What does PMS stand for in this project?" --include docs --pot local:cli-usage-eval
potpie resolve "What does PMS stand for in this project?" --include docs,resources --pot local:cli-usage-eval
```

The first and third retrieve the answer; the second returns four unrelated summary hits and no field-reference document. Its intentionally general summary does not mention the expansion, but the stored bytes do.

Recommendation: make the user-facing document filter cover summaries **and** passages, with optional expert controls for each. Until then, teach `--include docs,resources` as the normal document lookup and describe the distinction in flag help. A search miss should suggest the correct alternate path without requiring the agent to know the indexing architecture.

### 3. High: cross-family ranking can hide an exact answer below the display cutoff

An exact body token yielded a unique resource hit, but broad search placed it at rank 23. A clear rollback phrase placed the right section at rank 22. The text footer only says to use JSON, which expands these two responses to roughly 50–55 KB.

```sh
potpie search "ORCHID-731" --pot local:cli-usage-eval
potpie search "ORCHID-731" --include resources --pot local:cli-usage-eval
potpie search "disable Argo auto-sync after rollback" --pot local:cli-usage-eval
potpie search "disable Argo auto-sync after rollback" --include docs --pot local:cli-usage-eval
```

Recommendation: let query relevance and exact-match evidence dominate generic recency/authority boosts across families. Preserve strong source hits within the displayed budget. The definition path already successfully promotes a direct acronym expansion; ordinary exact-token retrieval needs equivalent protection.

### 4. High: the primary resolve workflow does not reliably retrieve mandatory preferences

`resolve "Add a new posting path to ledger-api safely"` returned 45 items but no testing preference. Explicit `--include coding_preferences` also returned none. The scope-only preference view returned the seeded rule; applying the task query filtered it out.

```sh
potpie graph read --subgraph decisions --view preferences_for_scope --scope service:ledger-api --limit 20 --pot local:cli-usage-eval
potpie graph read --subgraph decisions --view preferences_for_scope --scope service:ledger-api --query "Add a new posting path to ledger-api safely" --limit 20 --pot local:cli-usage-eval
```

The dedicated preference skill correctly teaches the scope-only workaround. However, the compact agent instructions promise `resolve` as the first read without ensuring relevant constraints survive that path. Retrieve applicable preferences independently of similarity to the task sentence, or return an explicit instruction to run the scoped preference read.

### 5. Medium: malformed inputs can look like valid negative evidence

Observed successful empty responses:

- `resolve rollback --include documents`: exit 0, zero items, `unsupported_includes` identifies the unsupported family. Text does warn; scripts that use exit status or item count can still misread this as a successful lookup.
- `graph read --subgraph infra_topology --view service_neighborhood --scope service:ledger-api --direction sideways`: exit 0, zero items, no invalid-direction error.
- `graph search-entities ledger-api --type service`: exit 0, zero entities; the known type is `Service`.

By comparison, unknown views and missing chunk IDs return actionable nonzero errors. Validate closed vocabularies consistently and return allowed values. Distinguish unknown types from a valid type with no matching entities.

### 6. Medium: result contracts expose useful evidence, but lack a consistent success and presentation model

Useful fields are present: claim IDs, provenance references, environment, chunk IDs, revisions, original text, and retrieval metadata. Batched `resource get` matched fixture text exactly before/after re-import and restart. These are strong building blocks for grounded answers.

The friction is how they are exposed:

- Graph commands wrap data under `result`; resolve/search have top-level `items`; resource reads have top-level `chunks`; records return a separate receipt. A generic CLI consumer needs multiple decoders. Our initial seed driver assumed a top-level `plan_id` and had to adapt.
- Failed commit verification produced exit **1**, outer `ok: true`, inner `status: committed`, and `verification.ok: false`. Keeping “write applied” distinct from “verified” is correct; the outer success field must make that distinction explicit and agree with the documented exit contract.
- Text `document_passages` prints `[?]`, IDs, and long decimal scores, omitting the label/snippet that the JSON already carries. Even `--detail full` did not provide a useful snippet. Show title, short source excerpt, match reason, and one ready-to-run fetch command.
- `confidence` sometimes reads high for a full page of only loosely related results; it is a coverage measure, not answer confidence. The negative passage probe did emit a useful weak-match warning (best similarity 0.143), which should be preserved and made consistent across query paths.
- `--query-threshold 0.7` removed all irrelevant passage candidates, but also removed the exact ORCHID token match (semantic similarity 0.356). That matches the explicit semantic-threshold help; it is **not** a universal relevance knob. The default hybrid behavior correctly kept the lexical hit.

Do not claim absent facts merely because retrieval returned zero rows, or factual support merely because `quality=ok` and rows exist. Unknown questions still returned unrelated candidates in these tests.

### 7. Medium: all skills can be “current” while their behavioral instructions are stale

All 11 installed Codex skills were current according to `skills list`. That only checks installed/template drift, not agreement with the running CLI.

| Shipped instruction | Observed behavior / correction |
|---|---|
| Three resource skills say a summary is the “only index”; markdown says there is no lexical fallback. | The hybrid chunk index found an exact token in a section with no summary. Teach both retrieval paths. |
| Resource import warns the document is unsearchable until summaries are written. | Its own response also says `index.indexed: true`; `search --include resources` succeeds. Qualify the warning as summary quality, not total searchability. |
| Graph skill says passage reads have no floor and always fill `--limit`. | Exact-token query with limit 5 returned one hit; relative filtering exists. |
| Graph skill says thresholds work only for preferences. | Passage thresholds are accepted, reported, and applied. |
| Graph skill says there is no `Adapter` node. | Seeded `Adapter` entities and `USES_ADAPTER` relations were accepted and retrieved. |
| Graph skill says repo-scoped active decisions return none. | A repo-anchored decision returned correctly with repo scope. Whether the fixture was linked to that scope is what matters. |
| Generated CLI skill says status reports the active pot rather than repo scope. | Current status resolves repo scope and exposes both effective and active pot fields. |
| Generated CLI skill lists `doctor` among commands accepting `--pot`. | Actual `doctor --pot …` fails with a usage error. |
| Setup/skills use `--agent`; plain status defaults to Claude and uses `--harness`. | After Codex setup, plain status claimed all Claude skills missing. `status --harness codex` correctly showed none missing. Teach this or persist the chosen harness. |

Relevant sources: [graph skill](../../../potpie/cli/templates/claude_plugin/skills/potpie-graph/SKILL.md), [markdown skill](../../../potpie/cli/templates/claude_plugin/skills/potpie-resource-markdown/SKILL.md), [PDF skill](../../../potpie/cli/templates/claude_plugin/skills/potpie-resource-pdf/SKILL.md), [spreadsheet skill](../../../potpie/cli/templates/claude_plugin/skills/potpie-resource-spreadsheet/SKILL.md), [resource import warning](../../../potpie/cli/commands/resource.py). The installed skill snapshots are included in the evidence bundle.

Recommendation: keep one short authoritative command contract, generate repeated command/flag vocabulary from it, and execute documented skill recipes against a fresh seeded pot. Tests need behavioral assertions (the expected evidence is visible), not only “command exits zero.”

### 8. Lower priority: resource discovery and setup need clearer recovery paths

`resource list` requires a known `--doc` slug; there is no document inventory at that entry point. An agent discovering an unfamiliar pot has to use graph entity search instead. Permit a document inventory without `--doc`, retaining the current section listing when supplied.

The first setup deliberately ran offline with a fresh model-cache path. It returned `ok: true` even though the non-hard `embeddings.model` step failed. Pointing `embedding_cache` at the already available model and restarting recovered retrieval. This was an induced cache miss, not evidence that normal online setup cannot download a model. Still, a successful setup header should state when semantic retrieval is not prepared. Before setup, `config set` also required a running daemon, complicating model-cache configuration.

## What passed beyond retrieval

- New pot creation, skill installation, source registration, semantic proposal/commit, resource import/linking, and single-fix record/readback worked, aside from the specifically documented seed defects.
- A timeline seed with an unsupported `Activity → Adapter` target was rejected with operation index and allowed endpoint types. Moving that reference to `mentions` allowed the plan. This was good, actionable validation.
- A decision plan requiring review was refused without approval and committed after the synthetic seed was reviewed and attributed to `harness:synthetic-eval`.
- Three document imports had summaries; the fourth intentionally had one pending summary. All four reported `graph.written: true`.
- Same-content re-import kept revision 1 and stable chunk IDs; it did not duplicate the chunks. No changed-content replacement or PDF extraction was tested.
- Batch fetch, Unicode, exact text fidelity, resource lookup failure, record persistence, count persistence, and cross-pot isolation passed. Resource fetch returned `resource_not_found` from the empty control pot; search there returned no data.
- Commit verification caught the write loss rather than silently declaring verified success. That mechanism is valuable even though its outer response needs clarification.

## Suggested order of work

1. Fix the environment claim collision and add backend conformance coverage.
2. Make document filtering intuitive and protect strong exact/document hits across family ranking.
3. Ensure resolve returns applicable preferences and make invalid queries fail clearly.
4. Reconcile the shipped skills with those behaviors; remove advice encoding obsolete limitations.
5. Normalize machine-readable outcomes and improve compact passage responses, then add document inventory.

The core surface can remain small: **resolve a task → search for evidence → fetch source text → record a learning**. Named graph views are useful expert follow-ups. Agents should not need to learn the graph/index split or a collection of ranking workarounds before those four steps work predictably.

## Reproduce and inspect

Use the frozen build and existing seeded pot from this workspace:

```sh
./.tmp/cli-usage-eval-20260910/potpie status --harness codex --pot local:cli-usage-eval
./.tmp/cli-usage-eval-20260910/potpie search ORCHID-731 --include resources --pot local:cli-usage-eval
./.tmp/cli-usage-eval-20260910/potpie resource get potpie://res/field-reference/recovery-marker/0000 --pot local:cli-usage-eval
```

Replay a selected query without writes:

```sh
python3 docs/evaluations/cli-usage-2026-09-10/replay.py --cli ./.tmp/cli-usage-eval-20260910/potpie --pot local:cli-usage-eval --case body-resources --output /tmp/potpie-read-replay.jsonl
```

Seed another fresh pot and run the 45 queries against a ready local CLI:

```sh
python3 docs/evaluations/cli-usage-2026-09-10/replay.py --cli ./.tmp/cli-usage-eval-20260910/potpie --pot local:cli-usage-replay --seed --output /tmp/potpie-fresh-replay.jsonl
```

The replay refuses to seed a reused pot and refuses to overwrite its transcript. It logs intentional errors and committed-but-unverified writes; its process completion is not a product pass verdict. Fixtures retain their 2026 dates, so the `30d` timeline case will naturally change if replayed later. It does not install skills, recreate the isolation harness, or repeat restart/control-pot tests.

The replay script was itself validated against another newly created pot: all fixture imports and the selected body-only retrieval completed, while the infrastructure collision reproduced again. These 20 additional invocations are separate from the 109-command evaluation count: [replay validation transcript](replay-validation.jsonl.gz). Ruff checks passed for the script. Import-path inspection confirmed all three product packages loaded from the freshly installed wheel target.

Evidence: [result index and hashes](results.json), [query matrix](queries.json), [raw transcript, gzip JSONL](transcript.jsonl.gz), [setup response](setup.json), [fixtures](fixtures), and [replay script](replay.py). Transcript entries contain full stdout, stderr, exit code, timing, and parsed payload. Named labels throughout this report locate the underlying observations.
