# Export and import a pot

Export a pot to a readable folder, then import it into a new pot:

```bash
potpie graph export ./pot-backup --pot local:my-project
potpie --host local pot create restored-project
potpie graph import ./pot-backup --pot local:restored-project
```

The path belongs to the machine running the CLI. The same commands work through
a local daemon or an updated managed host; the CLI transfers data, never asks a
remote server to open a client path. Both client and host must support snapshot
version 2. Restart a running local daemon after upgrading its code.

## Folder contents

| File | Contents |
|---|---|
| `manifest.json` | Format version, source pot ID, data filenames and resource file list |
| `entities.json` | Entity keys, labels, names, aliases and all other properties |
| `claims.json` | Claims, provenance/evidence, embeddings and validity timestamps |
| `resources/` | Original UTF-8 document chunks, manifests and retained historical revisions |
| `README.md` | A short description and import command |

JSON is indented and document chunks remain ordinary text files. An isolated
entity with no claims is included. Files are staged before publishing the export;
existing exports are protected unless you explicitly pass `--overwrite`:

```bash
potpie graph export ./pot-backup --pot local:my-project --overwrite
```

A destination ending in `.json` writes one JSON file instead of a folder. Import
also accepts the old version-1 JSON snapshots, which contain only claims and
labels; missing properties and document bytes cannot be recovered from an older
snapshot that never contained them.

## Restore behavior

Import validates the entire graph before changing it. It merges new entities and
claims into the selected pot, preserves unrelated data, and skips identical
records, so importing the same snapshot twice does not duplicate claims. When an
existing identity has different content, import fails instead of overwriting it.
Use a new pot for a clean restore or migration.

Canonical claim keys that contain the source pot ID are remapped to the selected
target. Stored claim evidence and entity identities are preserved. A successful
write advances the destination's own graph revision; source mutation receipts and
revision counters are not transplanted as destination execution history.

Document bytes are included by default. Resource manifests retain their revision
numbers, so evidence such as `potpie://res/manual/body/0000@rev2` still resolves
after restore. Existing target documents must be byte-identical. Resource files
are staged and rolled back if the graph import raises an error. A process crash
between publishing resource bytes and committing the graph can leave additional
unreferenced files; it does not remove existing evidence. Search indexes are
rebuilt after restore; an index failure is reported as a warning with a rebuild
command and does not discard restored data.

To deliberately omit document text:

```bash
potpie graph export ./graph-only --pot local:my-project --graph-only
potpie graph import ./pot-backup --pot local:restored-project --graph-only
```

With `--graph-only`, document references whose bytes are absent on the destination
cannot be opened until the documents are imported separately.

## Support and scope

Graph snapshots are supported by `in_memory`, `embedded`, `falkordb_lite`,
`falkordb`, and `neo4j`. The standard local resource store supports document
bundles. Stub backend profiles remain unsupported. Managed snapshot operations
require pot admin access and updated server dependencies; updating the CLI does
not upgrade a remote deployment.

A snapshot carries graph data and document evidence. It does not export login
credentials, pot membership, source registrations, mutation-plan history, inbox
items or host configuration. Resource bundles currently enforce the resource
transfer limit of 64 MiB of UTF-8 text; a managed host can impose a smaller HTTP
request limit. Oversized transfers are refused rather than silently truncated.

For a consistent backup, pause other processes that write to the same pot while
exporting. Graph data and document bytes are each read consistently, but the two
stores do not share a transaction across independent writers. The current format
loads the graph in memory; it is intended for transfers that fit the available
client and host memory.
