# Optional document fact extraction

`potpie document ingest` parses, chunks, stores, indexes, and links document structure without an LLM. Add `--extract-facts` to ask a user configured model for explicit entity relations supported by a quoted chunk passage.

For a one-time interactive setup, install the documents extra (`uv sync --extra documents` in this checkout, or `pip install 'potpie[documents]'`), start Ollama, pull Moondream (`ollama pull moondream`), then run:

```bash
potpie document setup
```

The command checks Docling, OCR, Ollama, and the chosen installed vision model's vision capability; asks whether to use picture captions and Gemini fact extraction by default; and prompts for the Gemini key with hidden input. Moondream is the default vision model, but setup accepts any installed Ollama model that reports vision support. It stores non-secret preferences in the local Potpie config and the key in Potpie's owner-only credential file. It does not ingest a document. An agent can run later ingests with the ordinary command, without passing the key or extra flags:

```bash
potpie document ingest design.pdf --doc design --pot YOUR_POT
```

Before setup, document ingestion remains zero-LLM by default. After setup, `--vision-provider none` and `--no-extract-facts` opt out for one ingest; explicit environment variables override saved model and endpoint settings. Rerun `potpie document setup` to change the defaults or replace the key.

```bash
potpie document ingest design.pdf --doc design --extract-facts
potpie graph inbox
potpie graph inbox --status rejected
potpie graph approve fact_...
potpie graph reject fact_... --reason "wrong entity"
potpie graph inbox clear --older-than 30d
potpie graph inbox prune
```

Without an API key, extraction uses local Ollama (`http://127.0.0.1:11434`, model `llama3.1`). If `POTPIE_FACTS_LLM_API_KEY` or `OPENAI_API_KEY` is set, extraction uses an OpenAI compatible API unless `POTPIE_FACTS_LLM_PROVIDER=ollama` overrides it. Set `POTPIE_FACTS_LLM_MODEL` for the model and `POTPIE_FACTS_LLM_BASE_URL` for a custom endpoint. No hosted credential is supplied by Potpie. `POTPIE_FACT_INBOX_TTL_DAYS` controls automatic closing of pending suggestions; the default is 30 days.

The extractor accepts only public ontology entity types and predicates. It checks that the quoted text exists in the cited revisioned chunk, validates cited pages against stored provenance, searches existing graph entities before choosing identities, and checks for conflicting live relations. It generates new identities automatically only for simple slugs such as services and datastores; types with scoped or external identities require review. A claim with confidence at least `0.9` that passes these checks is written to the graph and marked `verified` in the local SQLite fact Inbox. Other claims remain `pending` for review. An explicit approval can supply `--subject-key` and `--object-key` when identity search is ambiguous. The existing generic `graph inbox list` command continues to list graph work items; `graph inbox` without a subcommand lists document fact suggestions.

On a new document revision, pending suggestions from older revisions close as superseded. Rejected candidates remain in SQLite so extraction on the same revision does not suggest them again. `prune` deletes closed and rejected records after 90 days by default; use `--older-than` to change that window. Verified records remain available as an audit of automatic and approved writes.

The model's confidence is self reported. Exact quote and graph checks reduce unsupported writes, but a human should still inspect important claims in graph history. The claim evidence uses the immutable `@revN` chunk URI and page when available.

For PDFs, inspect `ingestion_diagnostics.picture_enrichment` in the ingest result. The counts distinguish detected figures, figures with OCR text, and figures with a visual caption; partial coverage is reported as a warning. OCR and small local vision models can misread labels or describe a diagram's topic without capturing arrow directions. For questions about a diagram's exact branches or order, check the cited page or exported figure before trusting the answer. An HTTP 429 or provider HTTP 5xx during fact extraction stops further model requests and reports a retryable error; the document and its searchable chunks remain ingested, but semantic fact extraction is incomplete. Re-run `--extract-facts` when the provider permits more requests.

`potpie resource get` now points to `potpie document show <chunk-uri>` for PDF chunks containing figures. That command returns `figure_artifacts` with local image paths and page numbers so a coding agent can inspect the actual diagram. Visual captions remain unverified hints. A fact whose quote is supported only by an image's OCR or caption is staged in the Graph Inbox even if the model reports high confidence; a user must inspect and approve it before it enters the graph.

Successful fact-extraction chunks are checkpointed locally by document revision, content hash, model, endpoint, and extraction prompt version. If the provider runs out of quota partway through, a later retry starts with the unfinished chunks. Changing the model or re-ingesting a new document revision triggers fresh extraction; superseded revision checkpoints are removed automatically.
