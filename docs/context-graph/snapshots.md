---
title: Graph snapshots
description: Export a pot's graph and document text to a readable folder, and merge it into another pot.
---

## Overview

> Status: reflects the typed-operation CLI, last reviewed 2026-10-06.

A snapshot is a portable copy of one pot: its entities, its claims and, by
default, the text of its documents. Export a pot to a folder, then import it
into a new pot:

```bash
potpie graph export ./pot-backup --pot my-project
potpie pot create restored-project
potpie graph import ./pot-backup --pot restored-project --yes
```

The path belongs to the CLI process. The CLI asks the engine for the snapshot
as data and writes the files itself; on import it reads and validates the files
and sends their contents. The daemon never opens a path on the caller's behalf,
the same rule `potpie resource import` follows ([resources.md](./resources.md)).
After upgrading Potpie, restart a running daemon (`potpie daemon restart`)
before exporting: a daemon from an older version refuses the new CLI.

## Folder contents

| File | Contents |
|---|---|
| `manifest.json` | Format version, source pot ID, data file names and the resource file list |
| `entities.json` | Entity keys, labels, names, aliases and every other property |
| `claims.json` | Claims with provenance, evidence, embeddings and validity timestamps |
| `resources/` | Original UTF-8 document chunks, manifests and retained revisions |
| `README.md` | A short description and the import command |

JSON is indented and document chunks stay ordinary text files. An entity with
no claims is included. The export is staged next to the destination and moved
into place at the end, and an existing destination is refused unless you pass
`--overwrite`:

```bash
potpie graph export ./pot-backup --pot my-project --overwrite
```

A destination ending in `.json` writes one JSON file instead of a folder.
Import also accepts version-1 JSON snapshots, which hold only claims and labels;
properties and document text that an older snapshot never contained cannot be
recovered from it.

## Restore behavior

Import validates the whole snapshot before it changes anything, and in a
non-interactive shell it needs `--yes`. It merges new entities and claims into
the selected pot, keeps unrelated data, and skips identical records, so importing
the same snapshot twice adds nothing. When an existing identity has different
content, import fails instead of overwriting it. Use a new pot for a clean
restore or migration.

Canonical claim keys that contain the source pot ID are rewritten for the target
pot. Claim evidence and entity identities are kept. A successful import advances
the destination's own graph revision; the source's mutation receipts and
revision counters are not carried over. Import is refused while commit history
capture is active on the target pot.

Document text is included by default. Resource manifests keep their revision
numbers, so evidence such as `potpie://res/manual/body/0000@rev2` still resolves
after a restore. A document that already exists in the target must be
byte-identical. Resource files are staged and rolled back if the graph import
raises an error. A crash between publishing resource bytes and committing the
graph can leave extra unreferenced files; it never removes existing evidence.
The resource search index is rebuilt after a restore; if that fails, import
reports a warning naming `potpie resource index rebuild` and keeps the restored
data.

To leave document text out:

```bash
potpie graph export ./graph-only --pot my-project --graph-only
potpie graph import ./pot-backup --pot restored-project --graph-only --yes
```

With `--graph-only`, document references whose text is absent from the target
cannot be opened until those documents are imported separately.

## Request versions

`graph export` and `graph import` use version 2 of the engine's
`export_snapshot`/`import_snapshot` operations:

- **Version 2** (what the CLI sends): export returns the snapshot as `payload`,
  with document text unless `include_resources` is false; import takes the
  snapshot as `payload`. Neither carries a path.
- **Version 1** (the default when a request names no version): export writes a
  graph-only snapshot to `destination` and import reads `source`, both paths on
  the machine that runs the engine. It remains for callers built against it.

A version-2 request that also names a path is refused, so a caller can't mix the
two.

## Support and scope

Snapshots are supported on `in_memory`, `embedded`, `falkordb_lite`, `falkordb`
and `neo4j`. Graph writes on FalkorDB and Neo4j are applied atomically against
the pot's revision. The local resource store carries document text. Stub backend
profiles report `not_implemented`.

A snapshot carries graph data and document evidence. It does not include login
credentials, source registrations, mutation-plan history, inbox items, commit
history or configuration. Document text is limited to 64 MiB of UTF-8 per
snapshot; larger transfers are refused, never truncated.

For a consistent backup, pause other writers to the pot while exporting. Graph
data and document text are each read consistently, but the two stores do not
share a transaction. The whole snapshot is held in memory, so it suits pots that
fit comfortably in the memory of the CLI and the daemon.
