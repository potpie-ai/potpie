---
name: potpie-resource-pdf
version: "1"
description: "Use when the user asks to ingest a PDF (report, contract, paper, manual, slide export) into Potpie so agents can search and cite it. Teaches the extraction-script flow: resolve the document's identity, split on the PDF's own structure into sections and ~4k-char chunks, import with `potpie resource import`, write retrieval-grade section summaries, and link the document with DOCUMENTS claims. Chunk text never passes through the agent's own output."
---

# Potpie Resource Ingestion: PDF

An ingested document is split in two. The **bytes** become pot-scoped chunk
files behind `potpie resource`; the **structure** becomes graph nodes — a
`Document` owning one `DocumentSection` per real division, joined by
`SECTION_OF`. Each section's summary becomes a claim and is the **only index
into its chunks**: retrieval quality is exactly summary quality. Ingestion is
harness-led — you read the source, choose the sections, and write the
summaries; Potpie validates, stores, and embeds. There is no Potpie-side
parser or scanner for PDFs, and no scan command writes the graph for you.

## Two rules that are never broken

1. **Chunk text never passes through your own output.** Write an extraction
   *script*; the script reads the PDF and writes the chunk files. Never type,
   paste, or echo document body text into a command argument, a summary, or a
   graph property. Reading chunks back afterwards (to summarize them) is fine —
   that is input, not output.
2. **Payloads never enter the graph.** The graph gets slugs, titles, summaries,
   hashes, and chunk ids. Import enforces the caps below and rejects rather
   than clamps.

## Step 1 — Resolve identity before extracting

```bash
potpie graph search-entities "q3 capacity review" --type Document --limit 10
```

If the document already exists, reuse its slug — importing the same `--doc`
replaces the content cleanly and bumps `revision`. Otherwise mint a slug:
lowercase letters, digits, single hyphens (`q3-capacity-review`). Never mint a
near-duplicate of an existing document slug; that fragments identity.

## Step 2 — Write and run the extraction script

The script emits one directory:

```
out/
  meta.json
  <section-slug>/0000.txt, 0001.txt, …   # zero-padded seq, UTF-8 text only
```

`meta.json`:

```json
{
  "source_ref": "file:///abs/path/q3-review.pdf",
  "source_kind": "pdf",
  "sections": [
    {
      "slug": "capacity-planning",
      "title": "3. Capacity Planning",
      "summary": "",
      "ordinal": 3,
      "content_hash": "sha256:6b6a…",
      "chunks": [
        {"seq": 0, "label": "3.1 Forecast model", "page": 12},
        {"seq": 1, "label": "3.2 Headroom targets", "page": 14}
      ]
    }
  ]
}
```

Hard limits, enforced at import:

| Field | Rule |
|---|---|
| chunk file | ≤ 8,000 chars hard cap; target ~4,000; split at paragraph boundaries |
| section | 1–5 chunks; more means the section is too coarse — split it into sub-sections |
| `label` | required per chunk, ≤ 200 chars — it is your only signal for picking a chunk later |
| `title` | ≤ 200 chars |
| `summary` | ≤ 2,000 chars; empty means `summary_pending` |
| `slug` | lowercase/digits/hyphens; **stable across re-imports** |
| `content_hash` | script-computed digest of the section's text; if empty, every re-import treats the section as changed |

Pick section slugs by *meaning* (`limitation-of-liability`), never by deriving
them from the heading text or its number — a retitled or renumbered heading
must not mint a new node and orphan the old section's claims.

### PDF specifics

- Check what is available before writing the script: `pdftotext` (poppler) on
  the PATH, or Python `pypdf`. Neither ships with Potpie — install into your
  own environment (`uv run --with pypdf`, a venv, or the system package), never
  into Potpie's.
- Split on the document's own structure, in order of preference: the outline /
  bookmarks (`pypdf` exposes them), numbered headings matched in the extracted
  text, then page ranges as a last resort. Never use a fixed-size sliding
  window — it cuts mid-argument.
- Record `page` on every chunk. It is the provenance a citation needs.
- If extraction yields almost no text, the PDF is scanned images. Stop and say
  so — OCR and binary payloads are out of scope for the resource store.

Script skeleton (adapt, then run it):

```python
import hashlib, json, sys
from pathlib import Path
from pypdf import PdfReader

TARGET = 4000  # hard cap is 8000; leave headroom

def pack(paragraphs):
    """Pack paragraphs into chunks of <= TARGET chars, never splitting one."""
    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > TARGET:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks

reader = PdfReader(sys.argv[1])
# 1. Extract text per page; keep (page_number, text) pairs.
# 2. Split into sections on the outline or numbered headings.
# 3. Per section: paragraphs -> pack() -> <slug>/0000.txt …
#    label each chunk from its leading heading or first line,
#    content_hash = "sha256:" + hashlib.sha256(section_text.encode()).hexdigest()
# 4. Write meta.json with summary "" for every section.
```

## Step 3 — Summaries: the only index

The script splits; it cannot judge. You write every summary by reading the
emitted chunk files (or `potpie resource get` after import). A summary is a
**retrieval card**, written for search rather than display: the terms,
synonyms, figures, and names a future searcher would type.

- Weak: `"Section about liability."`
- Strong: `"Limitation of liability: cap = 12 months of fees; carve-outs for
  IP infringement and data breach; indemnification obligations survive
  termination; governs disputes under NY law."`

Small document: fill every `summary` in `meta.json` before the first import.
Large document: import immediately with summaries empty (`summary_pending`),
then fill them in batches — edit `out/meta.json` and re-import the same
`--doc`. A section whose content is unchanged keeps the summary it already has
whenever the directory supplies none, so partial passes never blank earlier
work. Content nobody summarized stays effectively invisible: there is no
lexical fallback.

## Step 4 — Import and read the report

```bash
potpie resource import ./out --doc q3-capacity-review --source-ref "file:///abs/path/q3-review.pdf" --source-kind pdf
```

Import is atomic — a failure leaves any prior revision untouched. Pass `--pot`
when pot scope is ambiguous. Then check the report:

- `graph.written` must be true. Bytes land before graph state, so a rejected
  graph write means `resource get` works while **search finds nothing** — fix
  the reported issue and re-import.
- `sections_added / kept / changed / removed` — `changed` is the re-summarize
  list on a refresh.
- `summary_pending` — sections still invisible to semantic search.
- Errors carry stable codes: `resource_chunk_too_large` → fix the script's
  packing and re-run it; `resource_slug_invalid` → fix the slug;
  `resource_manifest_invalid` → fix `meta.json`. All are caller mistakes; the
  store was not down.

## Step 5 — Link it, or it is findable by luck alone

Import cannot guess what the document is *about*. A document with no
`DOCUMENTS` edge is reachable only by semantic search. Resolve the covered
entities with `graph search-entities`, then assert `DOCUMENTS`
(`Document`-or-section → any entity, subgraph `knowledge`):

```json
{
  "op": "assert_claim",
  "subgraph": "knowledge",
  "predicate": "DOCUMENTS",
  "subject": {"key": "document:q3-capacity-review", "type": "Document"},
  "object": {"key": "service:payments-api", "type": "Service"},
  "truth": "agent_claim",
  "confidence": 0.9,
  "description": "Q3 capacity review covers payments-api load forecasts, headroom targets, and scaling decisions for prod.",
  "evidence": [{"source_ref": "potpie://res/q3-capacity-review/capacity-planning/0000"}]
}
```

```bash
potpie --json graph propose --file documents-link.json
potpie --json graph commit mutation-plan:01XY --verify
```

Link a single section instead of the whole document when only that section
covers the entity.

## Step 6 — Verify retrieval, the way a future agent will

```bash
potpie search "headroom target prod"
potpie graph read --subgraph knowledge --view document_context --scope service:payments-api --limit 20
potpie resource get potpie://res/q3-capacity-review/capacity-planning/0000 --with-neighbors
```

A section hit already carries its chunk ids, so text is two calls away:
search, then one batched `resource get` (several ids in one call;
`--with-neighbors` covers a fact that spans a chunk boundary). If a phrase you
know is in the document does not surface a section, its summary is weak —
rewrite it and re-import.

## Refreshing and removing

Re-import the same `--doc` from a **complete** directory: a section missing
from the directory is treated as removed and its claim retracted. Stable slugs
plus `content_hash` keep the diff honest — only `sections_changed` needs
re-summarizing. `potpie resource rm q3-capacity-review --confirm` removes the
chunks and retracts the section claims; `resource list --doc <slug>` shows
what is stored.
