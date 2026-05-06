# RAG Failure Troubleshooting Guide

Potpie agents rely on RAG-style retrieval and long-context reasoning. When failures occur, they are typically not model-weight problems — they are structural issues across query formulation, embedding alignment, chunking strategy, vector search behavior, or execution ordering.

This guide defines 16 common structural failure modes, grouped into four categories, with a triage flow for each.

---

## How to use this guide

When an agent gives a wrong or unexpected answer, follow this triage flow:

```
Identify symptom → Map to failure mode → Apply targeted fix → Run verification test
```

Work through the categories in order. Most failures are resolved before reaching category 4.

---

## Category 1 — Retrieval Alignment Failures

These failures occur when the retriever returns the wrong documents, or fails to return the right ones, despite the index being correct.

---

### 01 — Query–document vocabulary mismatch

**Symptom:** Query uses different terms than the indexed content.

**Root cause:** The embedding space is not aligned between query phrasing and document language. Synonym or jargon gaps cause cosine distance to miss highly relevant docs.

**Fix:**
1. Expand query with HyDE or query rewriting before embedding.
2. Add domain synonym mapping to the retriever.
3. Re-embed documents using a domain-fine-tuned model.

**Verification test:** Manually inspect top-5 retrieved docs for 10 diverse queries. If relevant docs consistently rank 6–15, vocabulary drift is confirmed.

---

### 02 — Embedding model drift

**Symptom:** Retrieval quality drops after a model or index update.

**Root cause:** The query-time embedding model version differs from the one used at ingestion. Vector space geometry shifts — previously aligned embeddings become misaligned.

**Fix:**
1. Enforce embedding model version pinning in both ingest and query pipelines.
2. Re-embed the entire corpus after any model change.
3. Store the model hash in the index metadata and assert equality at query time.

**Verification test:** Run the same 20 benchmark queries before and after an index update. An MRR drop greater than 10% with identical queries confirms drift.

---

### 03 — Top-K cutoff too low

**Symptom:** The correct document exists in the index but never appears in answers.

**Root cause:** `k` is set so small that the relevant doc ranks just below the retrieval window. Especially common after corpus growth.

**Fix:**
1. Increase `k` temporarily (e.g. 20–50) and observe whether answer quality improves.
2. Add a reranker after retrieval to re-score a larger candidate pool.
3. Monitor rank distribution of ground-truth docs across a held-out eval set.

**Verification test:** Set `k=50` and check the rank position of the known-correct chunk for 10 failed queries. If it consistently appears at rank 6–25, the cutoff is the culprit.

---

### 04 — Sparse vs. dense retrieval mismatch

**Symptom:** Exact terms present in docs but retrieval still misses them.

**Root cause:** Dense-only retrieval cannot match on exact strings. Keyword queries (product codes, IDs, error messages) need BM25-style sparse retrieval.

**Fix:**
1. Switch to hybrid retrieval (BM25 + dense) using a fusion strategy such as RRF.
2. Route keyword-style queries to sparse retrieval and semantic queries to dense.
3. Inspect whether failed queries contain structured identifiers or precise phrases.

**Verification test:** Run 10 queries that contain exact strings present in the corpus. If BM25 alone retrieves them but dense retrieval fails, sparse retrieval is needed.

---

## Category 2 — Prompt / Reasoning Drift

These failures occur when the agent retrieves the correct documents but still produces a wrong or drifted answer.

---

### 05 — Context blending

**Symptom:** Agent merges details from multiple chunks into one incoherent answer.

**Root cause:** Multiple retrieved chunks are partially relevant. The model synthesizes a plausible-sounding but chimeric answer by stitching facts from different sources.

**Fix:**
1. Add source attribution in the prompt: "Answer only from DOCUMENT 1 or DOCUMENT 2, label which."
2. Reduce `k` to limit the number of retrieved chunks.
3. Use a reranker to surface only the single best chunk for factual queries.

**Verification test:** Ask a question where the answer appears in only one doc. Check if the model attributes to the correct document. If it cites a mix, context blending is active.

---

### 06 — Instruction anchoring failure

**Symptom:** Retrieved context overrides the system prompt persona or constraints.

**Root cause:** Long retrieved context crowds out the system prompt. The model attends more to the retrieved text than to instructions.

**Fix:**
1. Repeat critical instructions at the end of the prompt, after retrieved context.
2. Use a structured template that clearly separates system instructions from retrieved context.
3. Reduce retrieved chunk length so instructions remain a larger fraction of context.

**Verification test:** Insert a retrieved chunk that explicitly contradicts a system instruction. If the model follows the chunk over the instruction, anchoring failure is confirmed.

---

### 07 — Lost-in-the-middle degradation

**Symptom:** Correct chunk is retrieved but not used in the final answer.

**Root cause:** Transformer attention is weakest for content in the middle of a long context. Chunks placed at positions 3–6 in a 10-chunk context receive less weight than those at the start or end.

**Fix:**
1. Place the highest-ranked chunk first in the context, not last.
2. Trim total context to only the top 2–3 chunks for factual tasks.
3. Test positional sensitivity by swapping chunk order and observing answer change.

**Verification test:** Reorder retrieved chunks so the correct one is placed first, middle, and last. Score answer quality across positions. Significant middle degradation confirms this pattern.

---

### 08 — Hallucinated citations

**Symptom:** Model cites plausible but non-existent sources.

**Root cause:** The model was not grounded strictly to the retrieved set. It generates source-like strings from training memory rather than from the actual retrieved documents.

**Fix:**
1. Enforce citation grounding: "You may only cite sources that appear verbatim in the context. Do not generate any citations."
2. Post-process output to validate citations against retrieved chunk metadata.
3. Add a strict citation validation step in the pipeline.

**Verification test:** Collect model-cited sources from 10 responses. Cross-check each against retrieved chunk IDs. The citation rate outside the retrieved set quantifies hallucination frequency.

---

## Category 3 — Semantic Mismatch

These failures occur due to how documents are structured, chunked, or versioned in the index.

---

### 09 — Chunk boundary leakage

**Symptom:** Agent partially answers — the answer span is split across two consecutive chunks.

**Root cause:** Fixed-size chunking breaks documents mid-sentence or mid-paragraph. The critical fact lives at the boundary and no single chunk contains the full answer.

**Fix:**
1. Switch to semantic or paragraph-aware chunking.
2. Add a chunk overlap (e.g. 15–20% of chunk size) to preserve boundary context.
3. Identify which query types consistently receive split answers and tune chunk size for those document types.

**Verification test:** Locate 5 queries where the answer requires two sequential sentences. Inspect chunk boundaries in the index. If they fall between those sentences, boundary leakage is confirmed.

---

### 10 — Granularity mismatch

**Symptom:** Specific queries return vague answers; summary queries return fragment lists.

**Root cause:** Chunk size does not match query type. Fine-grained queries need small chunks; broad synthesis queries need large, section-level chunks. A single fixed size fails both.

**Fix:**
1. Use multi-granularity indexing: embed at both sentence and paragraph level, route queries by type.
2. Classify incoming queries as specific vs. broad and select the appropriate index.
3. Measure top-chunk length vs. query complexity correlation.

**Verification test:** Run 5 highly specific queries and 5 broad synthesis queries. Score whether chunk size returned matches query need. Consistently mismatched sizes indicate a single-granularity index.

---

### 11 — Semantic overlap confusion

**Symptom:** Multiple nearly-identical chunks retrieved; answer is inconsistent or averaged.

**Root cause:** The corpus contains near-duplicate content — versioned docs, paraphrased sections, or repeated boilerplate. Retrieval returns several that partially match; the model synthesizes an average.

**Fix:**
1. Deduplicate the corpus before indexing using semantic similarity thresholds.
2. Add document version metadata and filter at retrieval time.
3. Use a reranker to select the single highest-confidence chunk rather than passing all similar chunks to the model.

**Verification test:** Inspect retrieved chunks for 5 queries. If cosine similarity between top-3 chunks exceeds 0.92, deduplication is needed.

---

### 12 — Temporal staleness

**Symptom:** Answer is factually correct but outdated; model does not flag recency.

**Root cause:** The retrieved document is genuinely correct but reflects an older state. The model answers confidently without surfacing the document's age.

**Fix:**
1. Add a timestamp field to all indexed documents and include it in the retrieved context passed to the model.
2. Add a prompt instruction: "If a document date is present, note it and assess whether the answer may have changed."
3. Implement a freshness-weighted retrieval score.

**Verification test:** Index two versions of the same document (v1 and v2). Query for v2 content. If the model returns v1 facts without flagging staleness, temporal grounding is absent.

---

## Category 4 — Infrastructure / Deployment Ordering Issues

These failures appear after deployment and are often invisible in local benchmarks.

---

### 13 — Index / embedding version skew

**Symptom:** Agent performance drops post-deployment despite identical code.

**Root cause:** Embeddings were re-generated but the vector database still serves the old index. Query embeddings are computed with the new model but matched against old vectors.

**Fix:**
1. Implement an atomic index swap: build the new index fully before routing traffic to it.
2. Store the embedding model hash in the index header and assert it matches at query time.
3. Add a canary check: embed a known query and verify the top-1 retrieved doc against a fixture.

**Verification test:** After a deployment, retrieve the same 10 canonical queries. If top-1 results differ significantly from the pre-deployment baseline, version skew is present.

---

### 14 — Pipeline step misordering

**Symptom:** Retrieved context does not reflect the current query; answers seem generic.

**Root cause:** Retrieval is executed before query reformulation, user context injection, or conversation history assembly. The retriever sees an incomplete or different query than intended.

**Fix:**
1. Audit the step execution order. Retrieval should run after query enrichment (history injection, entity resolution, rewriting).
2. Log the exact string passed to the retriever vs. the raw user query.
3. Add an integration test asserting retrieval input format.

**Verification test:** Log the retriever's input for 10 live queries. Compare to the full enriched query the model receives. Any divergence confirms misordering.

---

### 15 — Reranker miscalibration

**Symptom:** Retrieval finds correct docs; reranker buries them below irrelevant results.

**Root cause:** The reranker was fine-tuned on a distribution that does not match production query patterns. It scores syntactic similarity over semantic relevance for the current domain.

**Fix:**
1. Audit the reranker's training data distribution against production query logs.
2. Collect production query–document relevance labels and fine-tune the reranker.
3. Run retrieval without the reranker and compare answer quality — if raw retrieval is better, the reranker is miscalibrated.

**Verification test:** Disable the reranker for an A/B slice of traffic. Compare NDCG@5 and answer accuracy. If raw retrieval outperforms reranked results, recalibration is required.

---

### 16 — Cold-start deployment gap

**Symptom:** Benchmarks pass locally; agent fails on first real queries post-deployment.

**Root cause:** Local benchmarks use warm caches, curated data, and pre-computed embeddings. Production hits cold paths: new query patterns, uncached embeddings, and live data freshness gaps.

**Fix:**
1. Add a cold-path integration test suite that clears all caches before running.
2. Use production query logs to seed the benchmark — do not rely solely on hand-crafted eval queries.
3. Implement a health-check endpoint that exercises the full live retrieval path.

**Verification test:** Clear all caches and run a fresh deployment against production traffic for 30 minutes. Compare failure rate to warm-cache benchmarks. A gap greater than 15% confirms cold-start sensitivity.

---

## Quick reference

| # | Failure mode | Category | Key fix |
|---|---|---|---|
| 01 | Query–document vocabulary mismatch | Retrieval alignment | Query rewriting / synonym mapping |
| 02 | Embedding model drift | Retrieval alignment | Version pinning + full re-embed |
| 03 | Top-K cutoff too low | Retrieval alignment | Increase k + add reranker |
| 04 | Sparse vs. dense mismatch | Retrieval alignment | Hybrid BM25 + dense retrieval |
| 05 | Context blending | Prompt / reasoning drift | Source attribution in prompt |
| 06 | Instruction anchoring failure | Prompt / reasoning drift | Repeat instructions after context |
| 07 | Lost-in-the-middle | Prompt / reasoning drift | Place best chunk first |
| 08 | Hallucinated citations | Prompt / reasoning drift | Strict citation grounding |
| 09 | Chunk boundary leakage | Semantic mismatch | Semantic chunking + overlap |
| 10 | Granularity mismatch | Semantic mismatch | Multi-granularity indexing |
| 11 | Semantic overlap confusion | Semantic mismatch | Corpus deduplication |
| 12 | Temporal staleness | Semantic mismatch | Timestamp metadata + freshness score |
| 13 | Index / embedding version skew | Infrastructure ordering | Atomic index swap |
| 14 | Pipeline step misordering | Infrastructure ordering | Audit + fix step order |
| 15 | Reranker miscalibration | Infrastructure ordering | Retrain on production distribution |
| 16 | Cold-start deployment gap | Infrastructure ordering | Cold-path integration tests |

---

*This guide is model-agnostic and infrastructure-agnostic. It does not require changes to model weights or new services — it provides a structured diagnostic layer before deeper refactors. Related issue: [#665](https://github.com/potpie-ai/potpie/issues/665)*
