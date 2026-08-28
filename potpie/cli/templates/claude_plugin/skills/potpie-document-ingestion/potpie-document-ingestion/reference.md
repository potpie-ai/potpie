# Document ingestion reference

## Architecture (two planes)

| Plane | Store | Contents |
|-------|-------|----------|
| Payload | `~/.potpie/resources/<pot_id>/<doc_slug>/` | `meta.json`, section `*.txt`, `elements.jsonl`, `*.prov.json`, `artifacts/` |
| Index | `registry.db` (SQLite FTS5) | Chunk text, OCR text, element + provenance tables |
| Graph | FalkorDB Lite | `Document`, `DocumentSection`, `DocumentElement` claims + chunk URIs |

Search merges graph section summaries with FTS chunk hits (`potpie search --include docs`).

## Supported formats

| Format | Extra | Parser | Provenance |
|--------|-------|--------|------------|
| `.md`, `.txt` | base `potpie` | stdlib heading split | none (`provenance_version=0`) |
| `.pdf` | `potpie[documents]` | Docling + RapidOCR | full (`provenance_version=2`) |
| `.pdf` degraded | pypdf in `[documents]` | `--allow-degraded` | none |
| `.png`, `.jpg`, … | `potpie[documents]` | RapidOCR + optional vision | file path + OCR |

## On-disk layout (after ingest)

```text
~/.potpie/resources/<pot_id>/<doc_slug>/
  meta.json                 # manifest: sections, provenance_version, parser_tier
  elements.jsonl            # Docling element ledger (PDF provenance path)
  artifacts/
    table-0001.csv          # table exports
    img-0001.png            # figure crops
  <section-slug>/
    0000.txt                # FTS-searchable chunk text
    0000.ocr.txt            # optional OCR overlay
    0000.prov.json          # chunk → element → page/bbox
```

Chunk URI: `potpie://res/<doc_slug>/<section_slug>/<seq>` (seq zero-padded, e.g. `0000`).

## Install

```bash
# Base: MD/TXT only
pip install potpie
# or uv tool install potpie

# PDF layout, OCR, provenance, images, tables, figures
pip install 'potpie[documents]'
# or: uv tool install --editable '.[documents]'  (from potpie repo root)
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `document service is not available` | `export CONTEXT_ENGINE_HOST_MODE=in_process` (daemon RPC not wired yet) |
| `PDF ingestion requires Docling` | `pip install 'potpie[documents]'` or pass `--allow-degraded` |
| Docling slow / first run hangs | First PDF downloads models; check `potpie doctor` documents tier |
| `file already imported` | Same PDF hash seen; use `--force` to re-ingest |
| Stale `page-*` sections in graph | Re-ingest with `--force` (retraction runs on section/element diff) |
| Empty figure text / `<!-- image -->` | Re-ingest with Docling path; figures get OCR + `artifacts/img-*.png` |
| Image caption missing | `--vision-provider local` needs Ollama; or `openai` + `OPENAI_API_KEY` |
| `invalid document slug` | Lowercase alphanumeric + hyphens only (`payments-oncall`, not `Payments_OnCall`) |

## Distinction: `potpie ingest` vs `potpie document ingest`

| Command | Purpose |
|---------|---------|
| `potpie ingest "note"` | Episodic ledger / graph notes (text episodes) |
| `potpie document ingest file.pdf` | Structured document payload + FTS + graph sections |

Do not use ledger ingest for PDFs or uploaded files.

## Deprecated CLI

`potpie resource *` mirrors `potpie document *` but prints a deprecation notice. Prefer `document` commands.

## Verification commands

```bash
potpie --json document list --pot <pot>
potpie --json document show potpie://res/<doc>/<section>/0000 --pot <pot>
potpie --json search "query" --include docs
```

Provenance block on `show` is `null` for legacy/degraded ingests; non-null lists `element_id`, `page_number`, `bbox`, `char_start`, `char_end`.
