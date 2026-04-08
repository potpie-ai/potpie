# Context Intelligence Layer — Testing Report

- Conversation ID: `019d2921-0431-760c-8242-378ebe0cfaaf`
- Project ID: `019d2402-b02f-7c2a-8174-2787ce8676d8`
- Provider: `HybridGraphIntelligenceProvider`
- Timeout budget: `4000ms`

## Run summary

- **Queries**: 10
- **Latency (ms)**: min=221, p50≈2197, max=4002

## Results

| Query | Coverage | Latency (ms) | Available | Missing | Evidence counts | Errors |
|------|----------|--------------|-----------|---------|----------------|--------|
| What happened in PR #694? | partial | 4002 | semantic_search, artifact_context, decision_context, discussion_context | change_history | sem=8 art=1 chg=0 dec=20 disc=1 own=0 | Resolution timed out after 4000ms |
| Why was this changed in PR #694? | partial | 4001 | semantic_search, artifact_context, decision_context, discussion_context | change_history | sem=8 art=1 chg=0 dec=20 disc=1 own=0 | Resolution timed out after 4000ms |
| Summarize the main rationale for merging PR #694. | partial | 4000 | semantic_search, artifact_context, decision_context, discussion_context | change_history | sem=8 art=1 chg=0 dec=20 disc=1 own=0 | Resolution timed out after 4000ms |
| Which files had review discussion in PR #694? | partial | 4001 | semantic_search, artifact_context, decision_context, discussion_context | change_history | sem=8 art=1 chg=0 dec=20 disc=1 own=0 | Resolution timed out after 4000ms |
| Who owns app/main.py? | complete | 221 | semantic_search, change_history, decision_context, ownership_context | - | sem=8 art=0 chg=10 dec=2 disc=0 own=5 | - |
| What PRs modified app/main.py? | complete | 330 | semantic_search, change_history, decision_context | - | sem=8 art=0 chg=10 dec=2 disc=0 own=0 | - |
| Find anything related to webhooks. | complete | 351 | semantic_search | - | sem=8 art=0 chg=0 dec=0 disc=0 own=0 | - |
| Where is the webhook handler implemented? | complete | 394 | semantic_search | - | sem=8 art=0 chg=0 dec=0 disc=0 own=0 | - |
| What was discussed in PR #999999? | partial | 4001 | semantic_search, decision_context | artifact_context, change_history, discussion_context | sem=8 art=0 chg=0 dec=20 disc=0 own=0 | Resolution timed out after 4000ms |
| Who owns nonexistent/path/that/does/not/exist.py? | partial | 352 | semantic_search | change_history, decision_context, ownership_context | sem=4 art=0 chg=0 dec=0 disc=0 own=0 | - |

## Effectiveness notes

- **PR #694 questions** return **partial evidence** within the 4s budget (semantic + artifact + discussions + decisions), even when one family times out.
- **Ownership/history** for `app/main.py` returned **complete** coverage quickly (owners, recent changes, decisions).
- **Semantic-only queries** (e.g. webhooks) return semantic hits without forcing structural families.

## Issues observed

- Neo4j emits warnings about missing `pr.number` property and missing `Fixes` relationship type. This is noisy but non-fatal; we should remove/guard those references in Cypher for cleaner ops.

## Documented responses (full payloads)

Captured **live** `resolve_context` output for the same 10 queries (same project, 4000ms timeout). The machine-readable file includes:

- `request`, `coverage`, `errors`, `meta` (including `schema_version`, `per_call_latency_ms`)
- `semantic_hits`, `artifacts`, `changes`, `decisions`, `discussions`, `ownership`

**File:** [019d2921-0431-760c-8242-378ebe0cfaaf-context-intelligence-responses.json](./019d2921-0431-760c-8242-378ebe0cfaaf-context-intelligence-responses.json)

**Truncation for readability:** long string fields are shortened with `…`; `semantic_hits` shows the first 3 rows; `decisions` the first 5; `changes` / `discussions` / `ownership` similarly capped. Where truncation applies, a `*_note` field records `showing N of M`.

### HTTP / API shape (same data)

If you call `POST /api/v1/context/query/resolve-context`, the JSON body matches:

```json
{
  "bundle": { "...": "full IntelligenceBundle as nested object" },
  "coverage": { "...": "same as bundle.coverage" },
  "errors": [ "...": "same as bundle.errors" ],
  "meta": { "...": "same as bundle.meta" }
}
```

### Example: PR #694 — excerpt (from captured run)

The first query returned **partial** coverage: semantic hits + PR artifact + decisions + one discussion row; `change_history` was empty within the budget (timeout noted in `errors`).

**Semantic hits (sample):** episodic facts such as “PR #694 added code coverage reporting…” and related `PART_OF_FEATURE` / `MODIFIED` edges.

**Artifact (`artifacts[0]`):** `kind: pr`, `identifier: "694"`, `title: "Fix/tests suite and run script"`, summary text derived from ingestion, `author: yashkrishan`, `extra.commits` with SHAs and messages.

**Discussions (`discussions[0]`):** `source_ref` like `PR #694 thread pr_conversation`, optional `file_path` / `line`, `headline` / `full_text` when present.

**Decisions:** up to 20 rows from `get_decisions` scoped to the PR; some rows are global review-bot text (see JSON for full text). PR-specific linkage appears where `pr_number` is set.

For **file-path queries** (`app/main.py`), the JSON shows non-empty `changes`, `ownership`, and a small set of `decisions`.

For **semantic-only** (“webhooks”), the JSON shows `semantic_hits` only and **complete** coverage for that plan.

For **invalid PR** (`#999999`), artifact/discussion are absent; semantic + broad `decision_context` may still return rows (noise vs. PR-specific — see JSON).

