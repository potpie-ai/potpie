---
name: potpie-document-ingestion
description: >-
  Ingests user documents (PDF, Markdown, plain text, images) into Potpie's
  context graph and searchable chunk store end-to-end. Handles layout PDFs with
  Docling provenance (page, bbox, tables, figures), RapidOCR, optional vision
  captions, FTS5 search, and graph projection. Use when the user uploads a
  file, asks to ingest/process/chunk/index a document, prepare docs for search,
  import a PDF handbook, extract tables or figures into the graph, or mentions
  .pdf .md .txt .png .jpg document ingestion. Not for episodic notes (use
  potpie ingest) or repo-wide source ingestion (use potpie-source-ingestion).
---

# Potpie Document Ingestion

End-to-end workflow for **file-based documents** into Potpie: one CLI command
parses, chunks, stores payloads on disk, indexes FTS5, and writes graph claims.
The agent orchestrates preflight, ingest, and verification — **do not** replace
this with ad-hoc Docling scripts, harness extraction scripts, or parallel DBs.

## Non-negotiables

- **Primary path:** `potpie document ingest` (one shot). Not `resource parse` +
  `resource import`, not custom Python chunkers, not `potpie ingest` for files.
- **PDF/images:** require `potpie[documents]` (Docling + RapidOCR). Fail loud if
  missing; only use `--allow-degraded` when the user accepts text-only PDF without
  provenance.
- **Do not use:** PyMuPDF, Tesseract-by-default, Unstructured/Marker, Kuzu, LanceDB,
  FastEmbed, or in-process torch VLMs. Potpie already routes vision via httpx → Ollama/OpenAI.
- **Host mode:** if document commands fail on daemon host, set
  `CONTEXT_ENGINE_HOST_MODE=in_process` and rerun.
- **Pot scope:** resolve pot before ingest (`potpie pot use` or `--pot`).
- **Doc slugs:** lowercase alphanumeric + hyphens (`api-guide`, not `API_Guide`).

## When to use (vs other skills)

| User intent | Skill / command |
|-------------|-----------------|
| Upload PDF/MD/image for search | **This skill** → `document ingest` |
| "Remember this note / meeting" | `potpie ingest` (ledger episodes) |
| Ingest entire repo / PR / tickets into graph | `potpie-source-ingestion` |
| Read existing project context | `potpie-agent-context` / `potpie search` |

## Quick start (happy path)

Copy this checklist and track progress:

```text
Document ingest:
- [ ] Phase 0 — Preflight (pot, doctor, host mode)
- [ ] Phase 1 — Resolve files + slugs
- [ ] Phase 2 — Ingest (document ingest)
- [ ] Phase 3 — Verify (list, show, search)
- [ ] Phase 4 — Report (+ optional context_record)
```

### Phase 0 — Preflight

```bash
potpie doctor
potpie --json pot list
potpie pot use <pot-id-or-name>    # if no active pot
```

Documents tier check (PDF/images):

```bash
potpie doctor    # inspect documents tier / Docling / Ollama recommendations
```

If `document` commands are unavailable:

```bash
export CONTEXT_ENGINE_HOST_MODE=in_process
```

Install gap (only when doctor or ingest reports missing Docling):

```bash
pip install 'potpie[documents]'
# from potpie repo: uv tool install --force --editable '.[documents]'
```

### Phase 1 — Resolve inputs

1. **Find files** — user path, workspace upload, or attachment. Supported:
   `.md`, `.markdown`, `.txt`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`,
   `.bmp`, `.tif`, `.tiff`.
2. **Choose `--doc` slug** — stable, short, descriptive (`payments-oncall`,
   `python-handbook`). One slug per logical document.
3. **Multi-file** — ingest each file with its own `--doc` (or same doc + `--force`
   only when intentionally replacing).
4. **Optional `--source-ref`** — canonical URI for citations (`file://…`,
   `https://…`, `confluence:page:123`).

### Phase 2 — Ingest (one command)

**Markdown / plain text** (base install):

```bash
potpie document ingest ./guide.md --doc api-guide --pot <pot>
```

**PDF with full provenance** (Docling — tables → CSV, figures → PNG + OCR):

```bash
potpie document ingest ./handbook.pdf --doc python-handbook --pot <pot>
```

**Re-ingest** (rebuild provenance, retract stale graph sections):

```bash
potpie document ingest ./handbook.pdf --doc python-handbook --pot <pot> --force
```

**Scanned / layout PDF without Docling** (degraded — user must accept no provenance):

```bash
potpie document ingest ./scan.pdf --doc scan --pot <pot> --allow-degraded
```

**Standalone image** (OCR + optional caption):

```bash
potpie document ingest ./diagram.png --doc arch-diagram --pot <pot> \
  --vision-provider local    # Ollama; or openai + OPENAI_API_KEY
```

Use `--json` globally for machine-readable reports:

```bash
potpie --json document ingest ./file.pdf --doc my-doc --pot <pot>
```

**Read the ingest report:**

- `graph_written` — FalkorDB claims applied
- `parser_tier` — `docling` vs `pypdf-degraded`
- `provenance_version` — `2` (full) vs `0` (legacy/degraded)
- `sections_removed` / `elements_removed` — graph retraction on re-import
- `errors` — stop and fix before claiming success

### Phase 3 — Verify

```bash
potpie --json document list --pot <pot>
potpie --json document show potpie://res/<doc>/<section-slug>/0000 --pot <pot>
potpie --json search "topic from document" --include docs
```

On `show`, inspect `provenance` for PDF traceability (page, bbox, element ids).
`provenance: null` means legacy or degraded ingest — searchable but not PDF-anchored.

Pick a section slug from `document list` output or `meta.json` sections (not
assumed `page-1` unless the parser produced that slug).

### Phase 4 — Report to user

Summarize:

- pot, doc slug, section count, `parser_tier`, `provenance_version`
- whether graph + search succeeded
- 1–2 example chunk URIs and a sample search hit
- if summaries are thin: note `summary_pending` and suggest editing section
  summaries in staging is optional (auto-summaries already power search)

Optional durable memory:

```bash
potpie ingest "Ingested python-handbook (161 sections, Docling provenance) into pot python-handbook — search with --include docs"
```

Or `context_record` via MCP when configured.

## Format routing (decision tree)

```text
User file
  ├─ .md / .txt     → document ingest (tier 0, no extra)
  ├─ .pdf
  │    ├─ potpie[documents] installed → Docling path (provenance v2, tables, figures)
  │    └─ not installed → install [documents] OR --allow-degraded (text-only)
  └─ image          → document ingest + [documents]; --vision-provider if captions needed
```

**Tables:** Docling exports `artifacts/table-*.csv` and embeds searchable table text in chunks.

**Figures:** `artifacts/img-*.png` + OCR text in chunk (replaces empty `<!-- image -->` placeholders).

**Do not** manually export Docling markdown and re-chunk unless debugging the parser.

## Common failures

| Symptom | Action |
|---------|--------|
| `PDF ingestion requires Docling` | Install `potpie[documents]` or add `--allow-degraded` |
| `file already imported` | `--force` to re-ingest same bytes |
| `document service is not available` | `CONTEXT_ENGINE_HOST_MODE=in_process` |
| Docling validation / empty sections | `--force` re-ingest; check PDF not corrupt |
| Vision caption empty | Ollama running locally or switch `--vision-provider openai` |
| Search finds nothing | Confirm `--include docs`; try FTS terms from chunk text |

Full troubleshooting: [reference.md](reference.md)

## Anti-patterns

- Writing `DocumentConverter` / `export_to_markdown` scripts as the **primary** ingest path
- Using `potpie resource parse` + `import` as default UX (deprecated; use `document ingest`)
- Using `potpie ingest` for PDFs (wrong subsystem)
- Assuming daemon-hosted ingest without checking host mode
- Skipping verification after ingest
- Inventing chunk URIs without `document list` / manifest section slugs

## Additional resources

- Pipeline architecture, on-disk layout, install: [reference.md](reference.md)
- Repo/source ingestion (not files): `potpie-source-ingestion` skill
- CLI auth/pot scope: `potpie-cli` / `potpie-pot-scope` skills
