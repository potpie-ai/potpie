---
title: Graph time and evidence
description: "Claim validity intervals, event versus observation time, and revision-pinned document citations."
---

## Overview

> Status: reflects the typed-operation CLI, last reviewed 2026-10-06.

Current graph reads use the claim's validity interval. A claim applies from its
start time, inclusive, until its end time, exclusive. Scheduling an end next
week leaves the claim current today. A read's `as_of` evaluates that same
interval at a requested time; audit reads can explicitly include invalidated
claims.

An Activity's `occurred_at` records when its event happened. `observed_at`
records when Potpie learned about it. Linking the Activity to another service
or repository later must not move the event into a different timeline window.
These guarantees do not reconstruct historical versions of arbitrary entity
properties.

New semantic writes normalize timestamps to UTC. Older native records whose
timestamp strings use other offsets still need an explicit data repair,
particularly on FalkorDB, which cannot convert those strings during a query.
Upgrading does not rewrite existing records.

## Document citations

An immutable citation includes its document revision:

```text
potpie://res/retry-runbook/recovery/0000@rev1
```

Resource discovery exposes versioned identifiers. Fetching an old identifier
returns that revision's original text, including when requesting neighbouring
chunks. A missing revision reports `resource_not_found`; it never falls back to
the current text. An old unversioned identifier becomes ambiguous after a
document changes and reports `resource_revision_ambiguous` instead of opening
replacement text. Use the versioned identifier returned by resource discovery.

Document revisions are retained across refreshes. Explicit `resource rm`
deletes the document's current and retained payloads. Reusing its slug after
deletion does not reuse an old revision number, so deleted citations remain
unavailable. Pot reset also preserves revision-counter metadata while removing
payloads. No automatic retention expiry is applied.

When evidence changes or disappears, resource operations identify affected
claims requiring review. Their conclusions are not automatically rewritten
from the replacement document. Reconciliation is an explicit graph correction
through the ordinary proposal and commit workflow.

An import publishes its pending evidence-review work with the new document
revision. A failed graph marker write is reported to the caller and remains
pending for an identical import retry. Completing an older revision's work
cannot clear a newer revision's pending review. Explicit deletion stops before
removing bytes if its review markers cannot be saved.

The marker remains visible in named graph reads, `resolve`/`search` text, and
`graph quality stale-facts`. These reads warn that the evidence needs review;
they do not claim the original conclusion has become false.

Source versions that were overwritten before revision retention existed cannot
be reconstructed. Retention covers the versions still present when the store
was upgraded and every later import. Repairing historical records remains a
separate, source-backed operation.

The resource store itself is described in [resources.md](./resources.md). See
[graph write guarantees](./graph-write-guarantees.md) for graph transaction,
correction targeting, and retry boundaries.
