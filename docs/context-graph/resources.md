---
title: Resource store
description: Where document payloads live, and how they are imported, indexed, found, read back, and removed with their pot.
---

## Overview

> Status: reflects the typed-operation CLI, last reviewed 2026-10-06.

The graph stores claims and pointers, never payloads (see [`vision.md`](./vision.md)).
The **resource store** is where a document's content lives instead. A document
splits into two halves:

- **Bytes** become chunk files on disk, scoped to a pot, behind one port
  (`ResourceStorePort`) so another storage backend can replace local disk.
- **Structure** goes into the graph: a `Document` entity owning one
  `DocumentSection` per real division of the source (heading, sheet, chapter). Each
  section's agent-written summary becomes a claim that normal retrieval finds.

A derived **retrieval index** over the chunk text lets a search reach a phrase no
summary mentions. Potpie does not parse documents: the coding agent writes an
extraction script that emits a chunk directory, and one `potpie resource import`
absorbs it, so chunk text never passes through the agent's output tokens. The
per-format skills `potpie-resource-pdf`, `potpie-resource-spreadsheet` and
`potpie-resource-markdown` teach that flow (see [`skills.md`](./skills.md)).

## Requirements

| # | Requirement |
|---|---|
| R1 | Payloads never enter the graph; the graph holds structure and pointers only. |
| R2 | The agent names a document, and its identity resolves through normal graph identity. |
| R3 | Documents split on their own structure into sections, and sections are searchable. |
| R4 | Every chunk fits one tool-call response, enforced at import. |
| R5 | Chunk text never passes through the agent's output tokens. |
| R6 | Import is atomic: a crashed run leaves no partially visible document. |
| R7 | Re-import publishes a new revision, keeps prior ones, and flags claims that cited changed text. |
| R8 | Everything is pot-scoped, and pot teardown removes documents with the graph. |
| R9 | `resource get` does no graph query and no embedding. |
| R10 | Chunk text is searchable even where no section summary mentions it. |
| R11 | An agent reaches chunk text in two calls (search, then get) and fetches several chunks in one. |
| R12 | Re-import re-summarizes only the sections whose content changed. |

## How it works

```mermaid
flowchart LR
    script["extraction script<br/>(written by the agent)"]
    dir[/"chunk dir<br/>meta.json + section/seq.txt"/]
    imp["potpie resource import"]
    disk[("home/resources/pot_dir")]
    idx[("retrieval index")]
    graph[("Document + DocumentSection")]
    search["potpie search"]
    get["potpie resource get"]

    script --> dir
    imp -->|"reads on the caller side"| dir
    imp -->|"1. bytes"| disk
    imp -->|"2. text"| idx
    imp -->|"3. structure"| graph
    graph -.->|"section summaries"| search
    idx -.->|"passages"| search
    search -->|"chunk ids"| get
    get -->|"file read"| disk
```

**Import.** The CLI reads the directory on the caller's side and ships its contents
as `files` (keyed by relative path), never a path. The operation never reads the
executing host's filesystem for a caller, so it behaves the same in-process and
behind the local daemon. The store validates slugs, sizes, and `meta.json`, writes
the revision to a staging directory, and renames it into place. The chunks then go
to the index, and the structure goes to the graph through the normal
semantic-mutation door, pre-approved as `resource_import` so a re-import's
retractions are not held for review.

Bytes land first because there is no cross-store transaction: a failed graph write
leaves orphan files the next import overwrites, while the reverse order would leave
claims citing missing chunks. An index failure is a warning, never a failed import;
`resource index rebuild` recovers it.

**Find, then fetch.** A search lands on a section's summary claim (its chunk ids
surface as `chunk_ids`) or on an index passage (chunk id plus document and section
keys). `resource get` resolves the id straight to a file.

**Re-import.** A re-import publishes a new current revision and keeps every prior
revision's bytes. The revision advances only when sections were added, changed, or
removed, or when chunk bytes or citation-visible metadata differ (even if the
extractor reused a stale `content_hash`); a byte-identical re-import is a no-op.
Claims that cited a changed or removed section are marked
`evidence_review_required`, not rewritten. A kept section keeps its prior summary
when the directory supplies none.

## Data model and contracts

A `Document` (`document:<slug>`, carrying `revision`, `source_ref`, `source_kind`,
and `section_count`) owns `DocumentSection` nodes (`docsection:<doc>:<section>`,
carrying `title`, `ordinal`, `summary`, and `chunk_count`) through `SECTION_OF`.
`DOCUMENTS` links a document or section to what it covers. Both live in the
`knowledge` subgraph with their own `documents` fact family. Chunks are files, not
nodes, so a 500-page PDF becomes tens of section nodes, not hundreds.

```
evidence id   potpie://res/<doc>/<section>/<seq>@rev<N>   immutable; seq zero-padded to 4 digits
legacy id     potpie://res/<doc>/<section>/<seq>          valid only while the document has one revision
chunk file    <home>/resources/<pot_dir>/<doc>/<section>/<seq>.txt
prior revs    <home>/resources/<pot_dir>/<doc>/.versions/<N>/
pot_dir       <sanitized pot id>-<sha256(pot id)[:16]>     injective; never escapes the root
chunk size    target 4,000 chars, hard cap 8,000           rejected at import, never clamped at read
summary       2,000 chars; title and label 200 chars      these become node properties (R1)
import        64 MiB per call, UTF-8 text only
get batch     at most 128 ids per call
```

`<home>` is `CONTEXT_ENGINE_HOME`, or `~/.potpie` when unset. The directory an
extraction script produces:

```
<dir>/meta.json   { source_ref, source_kind,
                    sections: [{ slug, title, summary, ordinal, content_hash,
                                 chunks: [{ seq, label, page?, offset? }] }] }
<dir>/<section>/0000.txt, 0001.txt, …
```

The agent supplies section slugs, so a retitled heading does not mint a new node.
`label` is required because it is the agent's only signal for picking among a
section's chunks. Sections should hold 1 to 5 chunks; import warns above that.
Summaries may be left empty (`summary_pending`) and written in a later pass, so a
large document becomes usable section by section.

## CLI and typed operations

Each command is one typed engine operation, so the in-process runtime and the local
daemon answer identically.

| Command | Operation | Kind |
|---|---|---|
| `resource import <dir> --doc <slug> [--source-ref] [--source-kind] [--pot]` | `resource_import` | context write |
| `resource get <id>... [--with-neighbors] [--full] [--pot]` | `resource_get` | read |
| `resource list --doc <slug> [--section] [--limit] [--full] [--pot]` | `resource_list` | read |
| `resource rm <doc> --confirm [--pot]` | `resource_rm` | destructive |
| `resource index status [--pot]` | `resource_index_status` | read |
| `resource index build [--doc] [--wait] [--pot]` | `resource_index_build` | index write |
| `resource index rebuild [--doc] --confirm [--pot]` | `resource_index_rebuild` | index write |
| `doctor` (`resources` and `resource_index` rows) | `resource_status`, `resource_index_status` | read |

`import` reports `sections_added`, `sections_kept`, `sections_removed`, and
`sections_changed` (what needs re-summarizing), plus a `graph` block (structure
written and read back) and an `index` block (profile, pending embeddings). With no
live `DOCUMENTS` claim on the document, the next action suggests linking it.

`get` returns each chunk's `resource_id`, `text`, `chars`, `revision`, `source_ref`,
and `requested` (false for a chunk `--with-neighbors` pulled in). Output is capped
at 32 KiB with an exact follow-up for anything omitted; `--full` lifts the cap, but
credential-like metadata stays redacted. A partial batch keeps the chunks that
resolved, adds one outcome per requested id, and exits 1.

`rm` marks claims citing the document for evidence review, retracts its section
claims, drops its index rows, then deletes the bytes of every revision. Without
`--confirm`, a `--json` or non-interactive call fails with
`destructive_confirmation_required`. `index rebuild` also needs `--confirm`, but
only because re-embedding is slow; nothing can be lost, so it is not destructive.

Store failures keep their own codes (`resource_chunk_too_large`,
`resource_not_found`, `resource_revision_ambiguous`, …) rather than a flat
`validation_error`, and all exit 1; see [`cli-flow.md`](./cli-flow.md).

## Retrieval index

| Profile | Behavior |
|---|---|
| `sqlite_hybrid` (default) | BM25 plus vectors, fused by reciprocal rank. Reports itself lexical, with the reason, when the vector extension will not load or no embedder is configured. |
| `sqlite_fts` | BM25 only, no embedder loaded. |
| `none` | No index; passage search returns `match_mode="disabled"`. |

The profile comes from `CONTEXT_ENGINE_RESOURCE_INDEX`, then the `resource_index`
config key (`potpie config set resource_index <profile>`, which validates the
value), then the default. An unknown profile does not take down other commands: it
degrades to a labelled `none` index that reports `ready=false` and the fix in
`resource index status`, `doctor`, and every import.

Import writes lexical rows inline and leaves vectors pending. A background drain
thread fills them in; only the daemon, or a CLI process serving engine operations
in-process, starts it. Pending work is a row state, so a killed process loses
nothing. `resource index build --wait` drains in bounded batches until nothing is
pending.

- `--include docs` searches section summaries; at the agent door (`potpie search`,
  `potpie resolve`) it also adds chunk text.
- `--include resources` searches chunk text only.
- Bare `potpie search` (intent `unknown`) covers both.

The graph views `knowledge.document_context` (which document covers a topic) and
`knowledge.document_passages` (which text says it) keep the split precise. Mixed
envelopes demote both families, so a document corpus cannot crowd out decisions or
prior bugs.

**Calibration.** The absolute similarity numbers (the 0.75 similarity blend and the
relevance confidence bands) were measured on `all-MiniLM-L6-v2` and apply only to
that model (`CALIBRATED_EMBEDDING_MODELS` in `core/ports/resource_index.py`). Every
other embedder gets rank-and-coverage relevance and no confidence band from
relevance, and an explicit passage `--query-threshold` is refused with
`resource_index_query_invalid`.

## Pot lifecycle

- `pot reset` and `pot archive` go through the typed reset operation. It purges the
  pot's documents and index rows only after the graph reset succeeded, and reports
  `resources_purged`: `true` or `false` from the store, `null` when no resource
  store is composed.
- While graph journal capture is active for a pot, reset and archive are refused
  with `journal_capture_active` before anything changes: they would discard the
  pot's commit history mid-generation, and no command retires a journal yet. The
  graph reset and the document purge each refuse on their own as well; the single
  check up front is what keeps a teardown from stopping between them. Document
  imports and removals on such a pot are journaled resource workflows: each one
  is a rollback barrier, and a restore re-checks every `potpie://res/` citation it
  would bring back against the store.
- `source remove` does not touch documents. A source row is not a key into the
  store (`source_ref` is a free-form URI); use `resource rm` or pot teardown.
- `source add` takes a closed kind table. Document kinds (`pdf`, `spreadsheet`,
  `markdown`, `csv`, …) exit 1 with `source_kind_is_a_document` and point at
  `resource import`; unknown kinds exit 1 with `unknown_source_kind`.

## Decisions

| Decision | Alternative rejected | Why |
|---|---|---|
| Sections are nodes; chunks are files | A node per chunk | Tens of nodes instead of hundreds; the section is the unit a human names. |
| Split on the document's own structure | Fixed-size sliding window | Headings and sheets are boundaries the author already chose. |
| Summaries as claims plus a derived passage index | Summaries only | Summaries carry judgment; the index catches text no summary mentions. |
| Bytes before graph state | Graph first | No cross-store transaction; bytes-first fails to harmless orphans. |
| Import ships `files`, never a path | Server reads a caller path | The executing host need not share the caller's working directory. |
| Retain immutable revisions | Rebind old ids to new bytes | A citation must keep opening the text it cited. |
| Reject oversized chunks at import | Clamp at read | Every stored chunk is safe to hand an agent. |

## Non-goals

No bundled PDF or spreadsheet parsers and no Potpie-run summarizer; no binary,
image, or audio payloads; no cross-pot sharing or deduplication. A graph
snapshot carries document text with the graph by default (`--graph-only` leaves
it out; see [snapshots.md](./snapshots.md)).

## Verification

```bash
potpie --json resource import ./out --doc q3-review --source-ref file:///q3.pdf
potpie --json graph neighborhood --entity document:q3-review --detail full  # revision, section_count
potpie --json search "liability cap" --include docs     # section claims carry chunk ids
potpie --json resource get potpie://res/q3-review/capacity/0000@rev1 --with-neighbors
potpie --json resource import ./oversized --doc big     # exit 1, resource_chunk_too_large
potpie --json resource index status                     # profile, match_mode, pending_embeddings
potpie --json resource rm q3-review                     # exit 1, destructive_confirmation_required
potpie --json pot reset --confirm                       # resources_purged: true
```

## Open questions

- **Retention is unbounded per document.** Re-import never garbage-collects cited
  bytes; `resource rm --confirm` is the only deletion boundary.
- **Late summaries do not reach `meta.json`.** A second-pass summary goes through
  the graph, so `resource list` can still show `summary_pending` for it.
- **Concurrent imports** are serialized per document with OS file locks; a remote
  object store would need an equivalent conditional write.

## See also

- [`ontology.md`](./ontology.md): the entity and predicate catalog.
- [`querying.md`](./querying.md): the read trunk and the agent envelope.
- [`writing.md`](./writing.md): the semantic-mutation write door.
- [`cli-flow.md`](./cli-flow.md): output contract, exit codes, destructive-command rules.
