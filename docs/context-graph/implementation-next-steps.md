# Context Graph Implementation Next Steps

Last reviewed: 2026-04-22 (after e2e run on a fresh pot)

Priorities below come from an end-to-end run: fresh pot, one raw decision
episode ingested via `potpie ingest --sync`, then `search` and
`POST /api/v2/context/query/context-graph` with both `goal=aggregate` and
`goal=answer`. The plumbing (events, reconciliation, provenance,
freshness, coverage, quality signals, `/status`) is solid. Value is
leaking at **extraction** and at the **semantic→structural bridge**.

## Where To Push Next, Ranked By E2E Impact

### 1. Extraction / ontology classifier

Single biggest blocker. From one 4-line decision the ingestion agent
produced 58 entities / 43 edges with:

- 22 of 34 canonical entities labeled `Feature` (`test`, `refactor`,
  `docs`, `operations`, `security workflows` all became `Feature` nodes
  with the same summary).
- 40 / 43 edges are `RELATES_TO`; only 3 are canonical (`FLAGS`).
- The actual decision (the 4-tool agent surface) is not a `Decision`
  node at all.

Until a plain decision becomes `Decision` + `DECIDED` / `SUPERSEDES`
edges instead of a mass of `Feature` + `RELATES_TO`, every downstream
recipe (`decisions`, `recent_changes`, `project_map`) returns empty
or noise.

Options:

- Stricter label inference pass after Graphiti extraction that maps
  extracted nodes to `CANONICAL_LABELS` using property presence and
  text cues.
- A second-pass deterministic classifier that rewrites drift labels /
  drift edges into ontology types before apply, with the drift version
  kept only as a fallback attribute.

### 2. Semantic→structural bridge for decisions and changes

`goal=answer` with an unscoped query (no `file_path`, `pr_number`,
service, or ticket) returns `decisions: 0` / `recent_changes: 0` and
`coverage.missing = [decision_context, change_history]` — even when
the graph has `Decision` nodes from prior ingests. The structural legs
are scope-gated.

When scope is empty, the decision / change legs should fall back to
semantic seeds (top-K semantic hits → their `source_node_uuid` /
`target_node_uuid` → structural lookup) rather than returning empty.
Same bridge used for causal-expand already exists; extend it to
`get_decisions` and `get_change_history`.

### 3. Entity canonicalization — shipped 2026-04-22

`domain/entity_canonicalization.canonicalize_reconciliation_plan` runs
at the head of `validate_reconciliation_plan`: trim + lowercase +
collapse internal whitespace, then merge duplicate entity keys (union
labels, first-seen-wins properties), rewrite edge / invalidation
endpoints, drop self-loops, and dedupe edges. Extension point:
`SYNONYMS` table in the same module. Merge count surfaces via
`plan.warnings`. Covered by
`tests/unit/test_entity_canonicalization.py`.

### 4. Answer synthesis — shipped 2026-04-22

`AnswerSynthesizerPort` added at `domain/ports/answer_synthesizer.py`.
`PydanticAIAnswerSynthesizer` (`adapters/outbound/synthesis/`) runs a
single-shot `pydantic_ai.Agent` call with `output_type=str` over a
bounded payload (decisions, recent_changes, project_map,
debugging_memory, ownership, fallbacks, source_refs) built by
`build_synthesis_payload`. Wired into `_answer_async` between
`resolve_context` and `bundle_to_agent_envelope`; bootstrap picks
the real synthesizer when `CONTEXT_ENGINE_ANSWER_SYNTHESIS_MODEL` is
set, else `NullAnswerSynthesizer`. Any failure (import, timeout,
network, validation) degrades to the count-string fallback and
stamps `meta.answer_summary_source="counts"`. Covered by
`tests/unit/test_answer_synthesis.py`.

### 5. CLI parity with the skill surface — shipped 2026-04-22

Four new Typer commands in `adapters/inbound/cli/main.py`:

- `potpie status [POT] [--intent … --repo … --file … --pr …]` →
  `POST /status`.
- `potpie resolve [POT] QUERY [--file … --services a,b --include …]`
  → `POST /query/context-graph` with `goal=answer`; plain mode prints
  `result.answer.summary`, `--json` returns the full envelope.
- `potpie overview [POT] [--repo …]` → `POST /query/context-graph`
  with `goal=aggregate`, `include=["graph_overview"]`.
- `potpie record --type … --summary … [--details '{…}' --source-refs
  pr:7,issue:12 --sync --idempotency-key …]` → `POST /record`.

All four reuse `_pot_id_or_git` / `_cli_client_or_exit` / `_flags`
just like `search` / `ingest`. Covered by
`tests/unit/test_cli_skill_commands.py`.

Follow-up (still open): skill recipes in `SKILL.md` should spell out
`goal=aggregate` explicitly so agents stop guessing `overview` /
`graph_overview` and hitting 422.

## Smaller UX Issues Worth Sweeping

- `potpie ingest --sync` returning `episode_uuid: null` — shipped
  2026-04-22. `DefaultIngestionSubmissionService._submit_agent_reconciliation`
  now pulls the first non-null uuid out of `ReconciliationResult.episode_uuids`
  and stamps it onto the sync `EventReceipt`, and also forwards `job_id`
  when the submission assigned one. The HTTP handler and
  `_receipt_to_run_result` already pass those fields straight through, so
  `potpie ingest --sync` no longer needs a follow-up `event show`.
- `potpie event list` actor/source columns — shipped 2026-04-22. Added
  a tab-separated header row and two new columns, `source_channel` and
  `source_system`, between the existing `kind` and `submitted_at` columns.
  `created_by` is not a field on `IngestionEvent`; `source_system`
  (`context_engine_raw`, `github`, `linear`, …) is the closest durable
  actor proxy we already persist.
- `potpie status` (shipped 2026-04-22 as part of #5 CLI parity) closes
  the readiness-envelope gap that previously required direct HTTP calls.
  `potpie conflict list` / `conflict resolve` remain for the narrower
  predicate-family conflict surface.

## What Already Works (Keep)

- Event ledger → reconciliation run → episode → edge stamping all
  exercise cleanly end-to-end.
- Every search row carries the 13-field provenance block
  (`pot_id`, `source_event_id`, `event_occurred_at`, `confidence`,
  `reconciliation_run_id`, …).
- `/status` returns the full readiness envelope an agent needs
  (source capability matrix, event-ledger health, reconciliation
  health, recommended maintenance jobs, freshness, open conflicts).
- Provenance contract tests pin the flat-to-nested Neo4j → API
  translation.
- Query planner → per-family executor registry → merged envelope
  (single-leg vs. multi-leg `kind="multi"`) works as designed.

## Non-Negotiables

- Do not add a public tool per context family.
- Do not make source-specific graph write branches.
- Do not copy full source payloads into the graph by default.
- Do not make agents know Graphiti, Cypher, or Neo4j labels.
- Do not return facts without enough provenance for the consumer to
  understand where they came from and how fresh they are.
