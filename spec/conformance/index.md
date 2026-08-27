---
id: SPEC-CONFORMANCE-INDEX
title: Context Runtime Conformance Index
kind: index
revision: 1
maturity: draft
owners:
  - team:potpie
depends_on: []
change_ref: null
---

# Context Runtime Conformance Index

This derived index is the navigation entrypoint for current Context Runtime
implementation and integration verification. Canonical claims and evidence
remain in the six conformance records linked below.

The current tree contains one stable record path for each defined conformance
scope. Git history stores prior record versions; dates, sequence numbers,
implementation versions, and pull-request numbers are not encoded in current
filenames.

## Current Records

The five module records cover all current module behavior scopes. Successor
Daemon and CLI records pin implementation
`1db96d660b87d5cf50398a37318e1dbbf704610e` and verify accepted Daemon revision
2, including `DAEMON-052` through `DAEMON-056`; the other three module records
retain their prior passed identities. The cross-system record remains stale
because its pinned implementation and live PR-head integration identity have
not yet advanced.

| Scope | Contract | Current record | Spec revision | Behaviors | Result |
|---|---|---|---:|---:|---|
| Context Engine | [SPEC-CONTEXT-ENGINE](../modules/context-engine.md) | [CONF-CONTEXT-ENGINE](context-engine.md) | 1 | 34 | passed |
| Potpie Resource Manager | [SPEC-POTPIE-RESOURCE-MANAGER](../modules/potpie-resource-manager.md) | [CONF-POTPIE-RESOURCE-MANAGER](potpie-resource-manager.md) | 2 | 37 | passed |
| Daemon | [SPEC-DAEMON](../modules/daemon.md) | [CONF-DAEMON](daemon.md) | 2 | 56 | passed |
| CLI | [SPEC-CLI](../modules/cli.md) | [CONF-CLI](cli.md) | 1 | 33 | passed |
| Potpie Capabilities | [SPEC-POTPIE-CAPABILITIES](../modules/potpie-capabilities.md) | [CONF-POTPIE-CAPABILITIES](potpie-capabilities.md) | 1 | 12 | passed |
| Cross-system | [SPEC-SYSTEM](../system.md) | [CONF-SYSTEM](cross-system.md) | 1 | 23 | passed |

The five module records contain their own behavior traces. The cross-system
record contains the `SYS-*` trace, links the module records, and owns evidence
that spans module boundaries.

## Current Integration Target

Cross-system conformance pins the exact pre-merge identity; there is no
separate PR-specific conformance record. The row below is a historical passed
result and derives stale because the local implementation commit is not yet the
live PR head.

| Pull request | Base ref and commit | Verified PR head | Implementation | Result |
|---|---|---|---|---|
| `potpie-ai/potpie#1057` | `main` at `20a8389cabec6e5924b1e3d4ef12d1dcfe900a3c` | `3e5edfd584aea53682720c3684e6fd78646fa1b3` | `a0b52654f6fed50ec790cc2a72ccd581611ed3be` | passed; stale |

[CONF-SYSTEM](cross-system.md) additionally records the synthetic merge
candidate, merge tree, exact CI runs, current review gate, accepted contract
refs, and linked module records. It does not claim that human review is
approved or that the PR has merged.

## Historical Lineage

Each current record identifies its immediately previous version by stable or
historical record ID, full Git ref, and repository path. The migration baseline
is commit `3e5edfd584aea53682720c3684e6fd78646fa1b3`:

| Current path | Previous record ID | Previous path at baseline ref |
|---|---|---|
| `context-engine.md` | `CONF-CONTEXT-ENGINE-2026-08-24-01` | `spec/conformance/context-engine-2026-08-24.md` |
| `potpie-resource-manager.md` | `CONF-POTPIE-RESOURCE-MANAGER-2026-08-24-01` | `spec/conformance/potpie-resource-manager-2026-08-24.md` |
| `daemon.md` | `CONF-DAEMON` | `spec/conformance/daemon.md` at `604c3eb5c9a561eec959ab688c279d04e9e6ff5b` |
| `cli.md` | `CONF-CLI` | `spec/conformance/cli.md` at `604c3eb5c9a561eec959ab688c279d04e9e6ff5b` |
| `potpie-capabilities.md` | `CONF-POTPIE-CAPABILITIES-2026-08-24-01` | `spec/conformance/potpie-capabilities-2026-08-24.md` |
| `cross-system.md` | `CONF-SYSTEM-2026-08-24-01` | `spec/conformance/cross-system-2026-08-24.md` |

Earlier August 21 versions remain at their original committed paths in Git
history. They do not remain as duplicate files in the current tree.

Historical examples:

```bash
git show 3e5edfd584aea53682720c3684e6fd78646fa1b3:spec/conformance/cli-2026-08-24.md
git show 012d3638f2eae62685ea2f711c9c7a7b0dfeae84:spec/conformance/cli-2026-08-21.md
git log --follow -- spec/conformance/cli.md
```

## Update Convention

Update a stable record only when a durable verification identity changes:

- accepted `spec_id`, `spec_revision`, or `spec_ref`;
- selected `implementation_ref`;
- in-scope behaviors or dependencies;
- reproducible evidence or aggregate conclusion; or
- the PR-head and base-commit pair used as an integration target.

A successor replaces the current contents at the same stable path and points
to the immediately previous record version through `previous_record_id`,
`previous_record_ref`, and `previous_record_path`. The legacy portable-validator
field `previous_record` remains `null` because the prior artifact intentionally
does not resolve in the current tree. The prior Git object is not altered. If
none of the durable identities changed, keep the routine result in CI rather
than publishing a new repository record.

A new conformance file is appropriate only when an accepted contract defines a
genuinely new independent scope. Cross-module integration stays in
`cross-system.md`; a PR, release, date, or repeated check does not create a new
scope.

Freshness is derived by comparing each record's pinned identities with the
selected current targets. It is never authored as contract or index status.
