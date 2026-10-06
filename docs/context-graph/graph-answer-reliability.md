---
title: Graph answer reliability
description: "What a graph read answer carries: bounded kind-specific details, verification outcomes, exact identity lookup, and replayable follow-up commands."
---

## Overview

> Status: reflects the typed-operation CLI, last reviewed 2026-10-06.

The graph read surface returns bounded, kind-specific details alongside the
claim and its evidence references. Fixes expose root cause, remedy steps,
verification status and individual check outcomes. Decisions expose their
rationale and rejected alternatives. Infrastructure neighbourhoods include
`EXPOSES` API links and the relevant endpoint details.

These details survive compact and full named reads, JSON output and human
output from `resolve` and `search`. Authored fields are selected explicitly;
internal entity properties and large evidence bodies are not copied into each
answer. Detail text is capped at 2,000 characters and detail lists at 12
entries. Omission metadata identifies truncated content. Follow-up commands open
the selected entity or source text so callers can inspect the complete stored
data.

## Verification outcomes

Verification outcomes affect ranking according to their meaning. Explicit
successful checks add corroboration; failed, partial, unknown and numeric
statuses do not. Older `VERIFIED` claims without an outcome keep their
historical positive meaning and are labelled `legacy_verified`, rather than
being displayed as a newly observed test pass.

## Identity lookup

Identifier lookup distinguishes exact matches, ambiguity across repositories,
missing exact identifiers and approximate candidates. Keep the repository scope
when asking for an unqualified PR or issue number. The response reports the
families searched, including families that returned no items. Additional-results
metadata describes retrieval limits; it does not prove that the question was
answered.

## Follow-up commands

Generated follow-ups quote identifiers for the shell and keep the selected pot.
Source-passage commands request neighbouring chunks
(`potpie resource get <id> --with-neighbors`). Existing immutable revision
references are preserved, so following an older citation uses the
source-version rules in [graph time and evidence](./graph-time-and-evidence.md).

These are read-time behaviours: they do not repair historical records. How
reads, budgets and partial results work in general is in
[querying.md](./querying.md) and [cli-flow.md](./cli-flow.md).
