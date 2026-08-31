# Document ingestion examples

## Single PDF (Docling + provenance)

```bash
export CONTEXT_ENGINE_HOST_MODE=in_process
potpie pot use python-handbook
potpie document ingest ./handbook.pdf --doc python-handbook --pot python-handbook
potpie document show potpie://res/python-handbook/<section>/0000
potpie search "async context managers" --include docs
```

## Re-import (retract stale graph sections)

```bash
potpie document ingest ./handbook-v2.pdf --doc python-handbook --pot python-handbook --force
```

## Office formats (requires potpie[documents])

```bash
potpie document ingest ./spec.docx --doc product-spec --pot my-pot
potpie document ingest ./deck.pptx --doc onboarding-deck --pot my-pot
potpie document ingest ./costs.xlsx --doc q1-costs --pot my-pot
```

## HTML

```bash
# With Docling (preferred when [documents] installed)
potpie document ingest ./export.html --doc wiki-page --pot my-pot

# Without Docling: stdlib HTML extractor (headings → sections)
```

## Multi-file batch

```bash
POT=my-pot
for f in docs/*.md; do
  slug=$(basename "$f" .md | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  potpie document ingest "$f" --doc "$slug" --pot "$POT"
done
```

## Confluence HTML export walkthrough

1. Export page as **HTML** from Confluence (or save rendered page).
2. Choose a stable slug: `payments-runbook` (lowercase, hyphens).
3. Ingest:

```bash
potpie document ingest ./Payments_Runbook.html \
  --doc payments-runbook \
  --pot payments-oncall \
  --source-ref "confluence:payments-runbook"
```

4. Verify search:

```bash
potpie document list --pot payments-oncall
potpie search "rollback procedure" --include docs
potpie document show potpie://res/payments-runbook/<section>/0000 --with-neighbors
```

5. If the export is a single blob without headings, expect one `body` section; re-export with structure or split manually.

## Images with OCR + caption

```bash
potpie document ingest ./architecture.png --doc arch-diagram --pot my-pot \
  --vision-provider local
```

## Record ingest in graph memory (MCP)

After a successful ingest, record a durable doc reference:

```bash
potpie record --type doc_reference --summary "Ingested python-handbook: 161 sections, parser_tier=docling, provenance_version=2; search with --include docs" --pot python-handbook
```
