---
name: potpie-resource-markdown
version: "1"
description: "Use when the user asks to ingest a markdown or HTML document — an exported doc, wiki or Confluence page, web page, runbook, spec, or RFC — into Potpie so agents can search and cite it. Teaches the extraction-script flow: split on the heading tree into sections and ~4k-char chunks, import with `potpie resource import`, write retrieval-grade section summaries, and link the document with DOCUMENTS claims. Chunk text never passes through the agent's own output."
---

# Potpie Resource Ingestion: Markdown / HTML

An ingested document is split in two. The **bytes** become pot-scoped chunk
files behind `potpie resource`; the **structure** becomes graph nodes — a
`Document` owning one `DocumentSection` per real division, joined by
`SECTION_OF`. Each section's summary becomes a claim and is the **only index
into its chunks**: retrieval quality is exactly summary quality. Ingestion is
harness-led — you read the source, choose the sections, and write the
summaries; Potpie validates, stores, and embeds. No Potpie-side parser or
scan command writes the graph for you.

**First decide whether to ingest at all.** Files inside the working repository
are already reachable by any agent with file tools, and repo understanding
belongs in the baseline flow (`potpie-repo-baseline`), not in chunk storage.
Ingest when the payload lives *outside* the tree — a web page, a wiki or
Confluence export, a vendor doc, a spec handed over as a file — or when the
user explicitly wants the document searchable as durable project memory.

## Two rules that are never broken

1. **Chunk text never passes through your own output.** Write an extraction
   *script*; the script reads the source and writes the chunk files. Never
   type, paste, or echo document body text into a command argument, a summary,
   or a graph property. Reading chunks back afterwards (to summarize them) is
   fine — that is input, not output.
2. **Payloads never enter the graph.** The graph gets slugs, titles, summaries,
   hashes, and chunk ids. Import enforces the caps below and rejects rather
   than clamps.

## Step 1 — Resolve identity before extracting

```bash
potpie graph search-entities "payments oncall runbook" --type Document --limit 10
```

If the document already exists, reuse its slug — importing the same `--doc`
replaces the content cleanly and bumps `revision`. Otherwise mint a slug:
lowercase letters, digits, single hyphens (`payments-oncall-runbook`). Never
mint a near-duplicate of an existing document slug; that fragments identity.

For a web page, fetch it with your own tools first and save the HTML or
extracted text to a local file; the URL becomes `--source-ref`.

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
  "source_ref": "https://wiki.internal/payments/oncall",
  "source_kind": "html",
  "sections": [
    {
      "slug": "rollback-procedure",
      "title": "Rolling back a bad deploy",
      "summary": "",
      "ordinal": 2,
      "content_hash": "sha256:41d0…",
      "chunks": [
        {"seq": 0, "label": "Preconditions and freeze checklist", "offset": 118},
        {"seq": 1, "label": "helm rollback steps + verification", "offset": 164}
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

Pick section slugs by *meaning* (`rollback-procedure`), never by deriving them
from the heading text — a retitled heading must not mint a new node and orphan
the old section's claims.

### Markdown / HTML specifics

- **Split on the heading tree.** Choose the heading level that yields sections
  of 1–5 chunks: usually `##` for a long markdown doc, `h2` for a page. A
  single flat page with no headings becomes one `body` section.
- **Never split inside a fenced code block** — a half command is worse than a
  long chunk. If a fence alone exceeds the cap, split at blank lines inside it
  and say so in the labels.
- **HTML: extract content, drop chrome.** Strip navigation, headers, footers,
  sidebars, and scripts; keep headings, paragraphs, lists, tables (rendered as
  text), and code. Python's stdlib `html.parser` suffices; a converter library
  is fine if already available — nothing here ships with Potpie.
- Record `offset` (source line number) on each chunk when you can — cheap
  provenance for citations.
- Markdown needs no parser dependency at all: headings are `^#{1,6} ` lines
  outside fences.

Script skeleton (adapt, then run it):

```python
import hashlib, json, re, sys
from pathlib import Path

TARGET = 4000  # hard cap is 8000; leave headroom

def pack(blocks):
    """Pack paragraph/fence blocks into chunks <= TARGET, never splitting a fence."""
    chunks, buf = [], ""
    for b in blocks:
        if buf and len(buf) + len(b) + 2 > TARGET:
            chunks.append(buf)
            buf = b
        else:
            buf = f"{buf}\n\n{b}" if buf else b
    if buf:
        chunks.append(buf)
    return chunks

text = Path(sys.argv[1]).read_text()
# 1. (HTML only) extract content text, dropping nav/boilerplate.
# 2. Split into sections at the chosen heading level, outside code fences.
# 3. Per section: blocks -> pack() -> <slug>/0000.txt …
#    label each chunk from its leading heading or first line,
#    content_hash = "sha256:" + hashlib.sha256(section_text.encode()).hexdigest()
# 4. Write meta.json with summary "" for every section.
```

## Step 3 — Summaries: the only index

The script splits; it cannot judge. You write every summary by reading the
emitted chunk files (or `potpie resource get` after import). A summary is a
**retrieval card**, written for search rather than display: the terms,
synonyms, commands, and error strings a future searcher would type.

- Weak: `"How to roll back."`
- Strong: `"Rollback procedure for bad payments deploys: freeze checklist,
  helm rollback to previous release, replay stuck webhooks; covers 'deploy
  broke checkout' and 'payments 5xx after release'; verification via
  dashboards + synthetic checkout."`

Small document: fill every `summary` in `meta.json` before the first import.
Large document: import immediately with summaries empty (`summary_pending`),
then fill them in batches — edit `out/meta.json` and re-import the same
`--doc`. A section whose content is unchanged keeps the summary it already has
whenever the directory supplies none, so partial passes never blank earlier
work. Content nobody summarized stays effectively invisible: there is no
lexical fallback.

## Step 4 — Import and read the report

```bash
potpie resource import ./out --doc payments-oncall-runbook --source-ref "https://wiki.internal/payments/oncall" --source-kind html
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
  "subject": {"key": "document:payments-oncall-runbook", "type": "Document"},
  "object": {"key": "service:payments-api", "type": "Service"},
  "truth": "agent_claim",
  "confidence": 0.9,
  "description": "On-call runbook covers payments-api: rollback, webhook replay, pager escalation, and dashboards.",
  "evidence": [{"source_ref": "potpie://res/payments-oncall-runbook/rollback-procedure/0000"}]
}
```

```bash
potpie --json graph propose --file documents-link.json
potpie --json graph commit mutation-plan:01XY --verify
```

Link a single section instead of the whole document when only that section
covers the entity. A runbook that fixes a known failure is also debugging
memory — consider recording the symptom → fix pair as the `potpie-debug-memory`
skill teaches, citing the section's chunk ids as evidence.

## Step 6 — Verify retrieval, the way a future agent will

```bash
potpie search "payments deploy rollback"
potpie graph read --subgraph knowledge --view document_context --scope service:payments-api --limit 20
potpie resource get potpie://res/payments-oncall-runbook/rollback-procedure/0000 --with-neighbors
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
re-summarizing. `potpie resource rm payments-oncall-runbook --confirm` removes
the chunks and retracts the section claims; `resource list --doc <slug>` shows
what is stored.
