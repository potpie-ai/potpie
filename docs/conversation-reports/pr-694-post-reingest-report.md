# PR 694 Post-Reingest Evaluation Report

- Project ID: `019d2402-b02f-7c2a-8174-2787ce8676d8`
- PR: `#694`
- Scope: validate improvements after structural reingest and bridge rebuild
- Date: `2026-03-26`

## Reingest execution summary

- Replay mode: full replay for `source_id=pr_694_merged` (ledger rows cleared, then reingested).
- Ingest result: `success`.
- Bridge counts:
  - `touched_by = 5`
  - `modified_in = 1`
  - `has_decision = 0`

## Questions asked and results

### Q1. Is PR #694 fully discoverable in PR context?
**Check**: `get_pr_review_context(project_id, pr_number=694)`

**Answer**: Yes.
- `found = true`
- `pr_title = "Fix/tests suite and run script"`
- `pr_author = "yashkrishan"`
- `pr_description` is present (non-empty)
- `commits` list is present with `4` commit entries
- `review_threads` count is `1`

**Verdict**: Improved significantly vs pre-reingest (previously no commits/author/description in the structured output).

---

### Q2. Do changed files have concrete diff evidence in graph now?
**Check**: `get_pr_diff(project_id, pr_number=694, limit=50)`

**Answer**: Yes, 5 file-level diff rows were returned:
- `app/modules/parsing/graph_construction/parsing_helper.py`
- `pyproject.toml`
- `scripts/run_tests.py`
- `scripts/run_tests.sh`
- `uv.lock`

Each row includes:
- `status`
- `additions`
- `deletions`
- `patch_excerpt`

**Verdict**: Major improvement (previously we only had summary-like evidence and no queryable patch snippets).

---

### Q3. Does change history still return empty for touched files?
**Check**: `get_change_history(project_id, file_path=<file>, limit=20)` for all 5 touched files.

**Answer**: No, it is no longer empty.
- `parsing_helper.py -> 20 rows`
- `pyproject.toml -> 17 rows`
- `scripts/run_tests.py -> 12 rows`
- `scripts/run_tests.sh -> 2 rows`
- `uv.lock -> 13 rows`

**Verdict**: Fixed. The `TOUCHED_BY` fallback plus ungated `MODIFIED_IN` now produce usable history.

---

### Q4. Can we provide code-level evidence for test/coverage claims?
**Check**: `get_pr_diff` excerpts on `pyproject.toml`, `scripts/run_tests.py`, `scripts/run_tests.sh`.

**Answer**: Yes.
- `pyproject.toml`: includes `pytest-cov` and coverage tool sections.
- `scripts/run_tests.py`: includes `--coverage/-c`, `--cov`, and HTML report wiring.
- `scripts/run_tests.sh`: usage text updated to include coverage option.

**Verdict**: Improved from inference-heavy to quoteable patch evidence.

---

### Q5. Do we now have richer "why" context for this PR?
**Check**: `get_pr_review_context` + `get_change_history`.

**Answer**: Yes.
- `pr_description` is available.
- `why_summary` for this PR appears in change history results.

**Verdict**: Better than pre-reingest where rationale came mostly from shallow PR summaries.

---

### Q6. Are review-thread decisions complete?
**Check**: `review_thread_count` and decision linkage.

**Answer**: Partially.
- `review_threads = 1` now visible in PR context.
- `has_decision = 0` in bridge stats for this replay.

**Verdict**: Better visibility in PR context, but code-node-level `HAS_DECISION` linkage remains sparse for this PR.

---

### Q7. Can we generate a participant roster from graph-only data?
**Check**: `pr_author`, review thread content, commit authors.

**Answer**: Partially.
- Author: available (`yashkrishan`).
- Commit-level participants: available via 4 commits.
- Reviewer/bot roster: still limited by available review comments in source data.

**Verdict**: Improved but still source-data constrained.

---

### Q8. Is this now good enough for tough PR-forensics style prompts?
**Check**: ability to answer with direct evidence table fields.

**Answer**: Mostly yes for:
- changed files,
- diff-level proof,
- commit-level context,
- author/description/time metadata.

Still limited for:
- deep review rationale where GitHub review/thread data is sparse,
- broad issue linkage (`Fixes`) when such edges are absent in graph.

**Verdict**: Marked improvement; not perfect completeness.

## Net impact of changes

## What clearly improved
- PR context is richer (`author`, `description`, `commits`).
- Diff evidence is now first-class (`get_pr_diff`).
- File-based history is no longer blank for this PR.
- Tough questions can now be answered with direct graph facts instead of only summary inference.

## What still remains
- `HAS_DECISION` node linkage can still be sparse depending on review data and line matching.
- If upstream GitHub discussion is thin, reviewer-rationale questions stay partially answerable.
- `Fixes`/issue linkage may be absent for PRs without explicit references.

## Suggested next validation pass

Run the same 8-question tough suite against:
- one PR with heavy review threads,
- one PR with multiple linked issues,
- one PR with large multi-file refactor,

to confirm these gains generalize beyond PR `#694`.
