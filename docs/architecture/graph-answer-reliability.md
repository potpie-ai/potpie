# Useful memory answers and exact identity lookup

The graph read surface returns bounded, kind-specific details alongside the
claim and its evidence references. Fixes expose root cause, remedy steps,
verification status and individual check outcomes. Decisions expose their
rationale and rejected alternatives. Infrastructure neighborhoods include
`EXPOSES` API links and the relevant endpoint details.

These details survive compact and full named reads, JSON output and human
output from `resolve` and `search`. Authored fields are selected explicitly;
internal entity properties and large evidence bodies are not copied into each
answer. Detail text is capped at 2,000 characters and detail lists at 12 entries.
Omission metadata identifies truncated content. Follow-up commands open the
selected entity or source text so callers can inspect the complete stored data.

Verification outcomes affect ranking according to their meaning. Explicit
successful checks add corroboration; failed, partial, unknown and numeric
statuses do not. Legacy `VERIFIED` claims without an outcome retain their
historical positive meaning and are labeled `legacy_verified`, rather than
being displayed as a newly observed test pass.

Identifier lookup distinguishes exact matches, ambiguity across repositories,
missing exact identifiers and approximate candidates. Callers should keep the
repository scope when asking for an unqualified PR or issue number. The
response reports the families searched, including families that returned no
items. Additional-results metadata describes retrieval limits; it does not
prove that the question was answered.

Generated follow-ups quote identifiers for the shell and retain the selected
pot. Source-passage commands request neighboring chunks. Existing immutable
revision references are preserved, so following an older citation uses the
source-version rules in [graph time and evidence](graph-time-and-evidence.md).

All of this changes local read behavior. It does not deploy a running host,
repair historical records, or establish comparative agent-quality results.
Those remain separate release, repair and evaluation work.
