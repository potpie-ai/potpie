# Phase 6: chunk-level embeddings (planned)

**Status:** Not implemented. Search today uses:

1. FalkorDB — section summary claim embeddings (graph semantic path)
2. SQLite FTS5 — chunk `content` + `ocr_text` (lexical path)
3. RRF merge in `DocsReader` when `include=docs`

## Target design

- After import, embed each chunk text via existing `EmbedderPort` / `build_embedder()` (no FastEmbed).
- Store vectors as claim-sidecar or dedicated chunk embedding claims in FalkorDB Lite (same mutation pipeline).
- Extend `DocsReader` RRF to include a third ranker: chunk dense similarity.
- Keep payloads on disk; graph holds URIs + embeddings only.

## Implementation checklist (future)

- [ ] `ResourceService` post-import hook: batch embed changed chunks
- [ ] Graph bridge: optional `CHUNK_EMBEDDED` or vector metadata on DocumentSection/chunk URI claims
- [ ] DocsReader: wire chunk vector search when embedder + vectors present
- [ ] Doctor gate: `chunk_embeddings_ready`
- [ ] Backfill command: `potpie document reindex-embeddings --doc <slug>`

Do not add LanceDB/Kuzu parallel indexes.
