---
name: potpie-resource-spreadsheet
version: "2"
description: "Use when the user asks to ingest a spreadsheet or CSV (cost report, inventory, tracker, export) into Potpie so agents can search and cite it. Teaches the extraction-script flow: one section per sheet, header-repeated row chunks, `potpie resource import`, summaries that carry the key figures, and — critically — deriving the durable facts as graph claims with chunk-id evidence, because chunked rows retrieve badly as text. Chunk text never passes through the agent's own output."
---

# Potpie Resource Ingestion: Spreadsheet / CSV

An ingested document is split in two. The **bytes** become pot-scoped chunk
files behind `potpie resource`; the **structure** becomes graph nodes — a
`Document` owning one `DocumentSection` per real division, joined by
`SECTION_OF`. Each section's summary becomes a claim and is the **only index
into its chunks**. Ingestion is harness-led — you read the source, choose the
sections, and write the summaries; Potpie validates, stores, and embeds. No
Potpie-side parser or scan command writes the graph for you.

**Spreadsheets have one extra rule.** Chunked rows match natural-language
queries badly: an ingest can look successful and silently answer nothing. The
*claims* must carry the derived facts — totals, key figures, per-entity
numbers, anomalies — with the chunks as backing evidence (Step 5). Skipping
that step is the most common way a spreadsheet ingest fails.

## Two rules that are never broken

1. **Chunk text never passes through your own output.** Write an extraction
   *script*; the script reads the workbook and writes the chunk files. Never
   type, paste, or echo cell contents into a command argument, a summary, or a
   graph property. Reading chunks back afterwards (to summarize or derive
   facts) is fine — that is input, not output.
2. **Payloads never enter the graph.** The graph gets slugs, titles, summaries,
   hashes, chunk ids, and the *derived* facts you judged durable. Import
   enforces the caps below and rejects rather than clamps.

## Step 1 — Resolve identity before extracting

```bash
potpie graph search-entities "vendor costs 2026" --type Document --limit 10
```

If the document already exists, reuse its slug — importing the same `--doc`
replaces the content cleanly and bumps `revision`. Otherwise mint a slug:
lowercase letters, digits, single hyphens (`vendor-costs-2026`). Never mint a
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
  "source_ref": "file:///abs/path/vendor-costs.xlsx",
  "source_kind": "spreadsheet",
  "sections": [
    {
      "slug": "saas-vendors",
      "title": "SaaS Vendors",
      "summary": "",
      "ordinal": 0,
      "content_hash": "sha256:9c1e…",
      "chunks": [
        {"seq": 0, "label": "rows 2-40: engineering tools, $310k/yr total"},
        {"seq": 1, "label": "rows 41-78: go-to-market tools, $120k/yr total"}
      ]
    }
  ]
}
```

Hard limits, enforced at import:

| Field | Rule |
|---|---|
| chunk file | ≤ 8,000 chars hard cap; target ~4,000; split on row boundaries only |
| section | 1–5 chunks; more means the section is too coarse — split it into sub-sections |
| `label` | required per chunk, ≤ 200 chars — it is your only signal for picking a chunk later |
| `title` | ≤ 200 chars |
| `summary` | ≤ 2,000 chars; empty means `summary_pending` |
| `slug` | lowercase/digits/hyphens; **stable across re-imports** |
| `content_hash` | script-computed digest of the section's text; if empty, every re-import treats the section as changed |

Those caps compound: a five-chunk section holds up to 40,000 characters and is
indexed by at most 2,000 of summary — roughly 20:1, and anything the summary
does not name is reachable only *after* search has already landed on that
section. On a sheet that is the norm rather than the exception, which is why
Step 5's derived claims are mandatory here: rows nobody named in a summary or a
claim are effectively unindexed.

Pick section slugs by *meaning* (`saas-vendors`), never by deriving them from
the sheet's display name verbatim — a renamed sheet must not mint a new node
and orphan the old section's claims.

### Spreadsheet specifics

- **One section per sheet.** A sheet too big for 5 chunks splits by its own
  logical grouping (a category column, a region, a quarter) into sub-sections
  like `saas-vendors-eng` / `saas-vendors-gtm`.
- **Repeat the header row at the top of every chunk**, so each chunk is
  self-describing when fetched alone. Render rows as compact TSV or pipe
  tables; drop empty columns.
- **Values, not formulas.** Extract computed values (`openpyxl` with
  `data_only=True`); note load-bearing formulas in the summary instead.
- **Labels carry the slice and its headline number** (`"rows 2-40: engineering
  tools, $310k/yr total"`), because an agent picks chunks by label alone.
- CSV needs only the stdlib `csv` module; `.xlsx` needs `openpyxl`, which does
  not ship with Potpie — install into your own environment (`uv run --with
  openpyxl`, a venv), never into Potpie's.

Script skeleton (adapt, then run it):

```python
import csv, hashlib, json, sys
from pathlib import Path

TARGET = 4000  # hard cap is 8000; leave headroom

def pack_rows(header, rows):
    """Pack rendered rows into chunks <= TARGET chars, header repeated."""
    chunks, buf = [], header
    for row in rows:
        line = "\t".join(row)
        if len(buf) + len(line) + 1 > TARGET:
            chunks.append(buf)
            buf = f"{header}\n{line}"
        else:
            buf += f"\n{line}"
    chunks.append(buf)
    return chunks

# 1. Read each sheet (csv.reader / openpyxl with data_only=True).
# 2. Per sheet: render header + rows, pack_rows() -> <slug>/0000.txt …
#    label each chunk with its row range and what those rows hold,
#    content_hash = "sha256:" + hashlib.sha256(sheet_text.encode()).hexdigest()
# 3. Write meta.json with summary "" for every section.
```

## Step 3 — Summaries: the only index

The script splits; it cannot judge. You write every summary by reading the
emitted chunk files (or `potpie resource get` after import). A summary is a
**retrieval card**, written for search rather than display — and for a
spreadsheet it must carry the *numbers and names* a future searcher would ask
for, since the row text underneath will not match their phrasing.

- Weak: `"List of SaaS vendors."`
- Strong: `"SaaS vendor costs FY2026: 77 vendors, $430k/yr total; top spend
  Datadog $96k, GitHub $54k, Vercel $31k; engineering $310k vs go-to-market
  $120k; renewals cluster in September."`

Small workbook: fill every `summary` in `meta.json` before the first import.
Large workbook: import immediately with summaries empty (`summary_pending`),
then fill them in batches — edit `out/meta.json` and re-import the same
`--doc`. A section whose content is unchanged keeps the summary it already has
whenever the directory supplies none, so partial passes never blank earlier
work.

## Step 4 — Import and read the report

```bash
potpie --json resource import ./out --doc vendor-costs-2026 --source-ref "file:///abs/path/vendor-costs.xlsx" --source-kind spreadsheet
```

Import is atomic — a failure leaves any prior revision untouched. Pass `--pot`
when pot scope is ambiguous. Run it with `--json`: the fields below are the
`--json` report, and human mode prints a condensed summary instead. Then check
it:

- `graph.written` must be true. It is readback-backed, not status-backed — a
  write the graph accepted but a read cannot see reports `false` and names the
  `missing_claim_keys`. Bytes land before graph state, so a rejected graph
  write means `resource get` works while **search finds nothing** — fix the
  reported issue and re-import.
- `sections_added / kept / changed / removed` — `changed` is the re-summarize
  list on a refresh.
- `summary_pending` — sections still invisible to semantic search.
- `recommended_next_action` — the one thing still missing, in order: a summary
  to write, a `DOCUMENTS` link when the document has none live (Step 5), or the
  retrieval check in Step 6.
- Errors carry stable codes: `resource_chunk_too_large` → fix the script's
  packing and re-run it; `resource_slug_invalid` → fix the slug;
  `resource_manifest_invalid` → fix `meta.json`. All are caller mistakes; the
  store was not down.

## Step 5 — Derive the facts (the step that makes a spreadsheet useful)

Decide which facts in the data are durable project memory — a cost that
constrains a decision, an ownership mapping, an inventory fact, a dated event
— and write them as ordinary graph claims exactly as the `potpie-graph` skill
teaches: check `graph catalog`, resolve entities with `graph search-entities`,
author operations, then propose and commit. The one non-negotiable is
**evidence**: every derived fact cites the chunk ids it came from, so the
claim stays auditable after the sheet changes:

```json
"evidence": [{"source_ref": "potpie://res/vendor-costs-2026/saas-vendors/0000"}]
```

Also assert `DOCUMENTS` (subgraph `knowledge`) from the document — or a single
section — to each entity the data covers, so the structural read finds it:

```json
{
  "op": "assert_claim",
  "subgraph": "knowledge",
  "predicate": "DOCUMENTS",
  "subject": {"key": "document:vendor-costs-2026", "type": "Document"},
  "object": {"key": "service:payments-api", "type": "Service"},
  "truth": "agent_claim",
  "confidence": 0.9,
  "description": "FY2026 vendor cost sheet covers payments-api third-party spend: Stripe, Datadog, PagerDuty line items and renewal dates.",
  "evidence": [{"source_ref": "potpie://res/vendor-costs-2026/saas-vendors/0001"}]
}
```

```bash
potpie --json graph propose --file derived-facts.json
potpie --json graph commit mutation-plan:01XY --verify
```

A document with no `DOCUMENTS` edge and no derived claims is findable by
semantic luck alone.

## Step 6 — Verify retrieval, the way a future agent will

```bash
potpie search "datadog renewal cost"
potpie graph read --subgraph knowledge --view document_context --scope service:payments-api --limit 20
potpie resource get potpie://res/vendor-costs-2026/saas-vendors/0000 --with-neighbors
```

A section hit already carries its chunk ids, so text is two calls away:
search, then one batched `resource get`. Bare `search` already includes
document sections, and mixes them with project memory the way a future agent
will see them. Test with the *question* a user would ask ("how much do we pay
Datadog"), not with the sheet's own column words — if it does not surface, the
summary or the derived claims are too thin.

`--include docs` narrows the envelope to documents alone, which splits the two
causes apart: a hit there but not in the bare search means the index works and
was simply outranked; nothing there means the summary and claims need the
rewrite.

## Report back

Report the ingest as commands and counts: the `resource import` you ran, the
`sections_created` / `sections_changed` / `chunks` it reported, the `plan_id`
from the derived claims and the DOCUMENTS link, and the retrieval check from
step 6 with the query you used. The derived facts are the deliverable here, so
name them and the chunk ids they cite — that pairing is what a reader has to be
able to audit.

Show the commands, never the chunk text. Pasting rows back into your reply routes
the spreadsheet through your own output, which is exactly what the extraction
script exists to prevent, and chunked rows read badly anyway — that is why the
facts were derived as claims in the first place.

No diagram. Figures belong in a markdown table with their units and period, where
they can be compared; mermaid cannot hold a number the reader wants to check.

## Refreshing and removing

Re-import the same `--doc` from a **complete** directory: a section missing
from the directory is treated as removed and its claim retracted. Stable slugs
plus `content_hash` keep the diff honest — only `sections_changed` needs
re-summarizing, and only its derived claims need revisiting.
`potpie resource rm vendor-costs-2026 --confirm` removes the chunks and
retracts the section claims; `resource list --doc <slug>` shows what is
stored.
