# Retrieval tuning: what was measured, and what it changed

**Date:** 2026-08-12 · **Scope:** `--include resources` — the retrieval index
over stored document chunks. Nothing here touches claim retrieval, which has its
own ranker and its own `match_mode` vocabulary.

The constants this doc justifies live in
`context-core/src/potpie_context_core/ports/resource_index.py`
(`SIMILARITY_BLEND`, `RRF_K`, `LEXICAL_RANK_DECAY`) and
`context-engine/.../resources/index/sqlite_fts.py` (`BM25_COLUMN_WEIGHTS`, the
prefix rule in `fts_match_expression`).

---

## What it was measured against

| | |
|---|---|
| corpus | 26 documents, 316 chunks, 1,257 embed windows, ~590k chars |
| composition | 20 real engineering documents (potpie's own `docs/` tree, split at its heading tree) plus 6 synthetic documents carried from the previous key |
| questions | 190 labelled `(document, section)` answers, stratified by kind |
| unanswerable | 12 probes, each verified absent from the corpus by word-boundary search |
| embedder | `sentence-transformers/all-MiniLM-L6-v2`, 384 dims |

Harness and fixtures: `pie/services/context-graph/benchmarks/retrieval/`.

The previous round tuned against 5 documents and 42 chunks, and **that turned
out to be the reason its conclusions were wrong**. At 42 chunks the semantic
arm's 120-candidate pool is larger than the corpus, so it returns *everything*:
measured over 19 questions, the lexical arm contributed **zero** candidates the
semantic arm had not already returned. Reciprocal-rank fusion under those
conditions is not doing the job it was written for, and every conclusion about
fusion drawn there was an artifact. At 316 chunks the arms disagree properly —
42.5 lexical-only and 67.6 semantic-only candidates per query.

---

## The method

Two stages, because an end-to-end run is ~11 minutes for 202 queries and a
parameter search needs hundreds of evaluations.

1. **Dump the arms once**, inside the service container: each query's lexical
   ranking, semantic ranking with cosine, and per-candidate term coverage.
2. **Replay offline**: fusion, the relevance floor, `_relevance`, the ranker's
   weighted mean, and the ×0.6 cross-include demotion are all arithmetic over
   that dump.

The replay is only trustworthy because it is checked: it reproduces a real
end-to-end result file query by query, rank and score, before any sweep is run.
On the final configuration the offline prediction and the live service agreed on
every metric to three decimals.

Configurations were chosen on a **stratified training half** and reported on a
**held-out half**, because a grid of 144 configurations over 190 questions will
find noise if allowed to.

---

## What moved, and what did not

Swept one knob at a time on the full 190-question key.

| knob | shipped | result |
|---|---|---|
| `RRF_K` | 60 | **inert, and lowering it hurts.** k∈{1…150}: MRR 0.727–0.741. k=60 is already at the top; k=20 costs 0.007 MRR and 0.021 top-1 |
| arm weights | 1.0 / 1.0 | **inert.** Parity is the best point measured; 2.5× semantic costs 0.019 MRR |
| relevance floor | 0.5 / 0.05 | **completely inert** — every value tested gave an identical score. Almost every hit is a lexical hit, and lexical hits are floor-exempt |
| candidate pool | 5×/50 | inert above 5×; shrinking to 2× costs 0.021 MRR |
| `LEXICAL_RANK_DECAY` | 10.0 | inert nearby; 4.0 and 20.0 are both worse |
| **cosine in `_relevance`** | not used | **+0.034 MRR, +0.036 r@5, +0.079 AUC** |
| **BM25 column weights** | flat | +0.007 MRR, and junk scores drop 0.476 → 0.429 |
| **prefix matching** | none reached SQLite | +0.006 MRR, +0.016 found |

### Why `RRF_K` was the wrong suspect

The previous round's headline hypothesis was that `RRF_K = 60` is a TREC-scale
constant flattening both arms on a tiny corpus. The arithmetic in that argument
is correct and the conclusion still does not follow, for a reason that sits one
layer down: **`_relevance` discards the fused order.**

A lexical hit's relevance is computed from `hit.lexical_rank` — the lexical
arm's *own* rank — not from its fused position. The ranker then sorts by that.
So fusion decides only which 24 chunks survive into the reader, and the order
within them is re-derived from one arm. Changing `k` reorders a list that is
about to be re-sorted anyway.

That also explains the previous round's other puzzle — that swapping the hashing
embedder for MiniLM moved cosine values a great deal and ranks not at all. The
cosine was not in the ranking. It is now.

---

## What changed

### 1. Cosine enters `_relevance` — the change that carries the gain

When both arms scored a chunk, relevance is now

```
(1 - SIMILARITY_BLEND) * (rank_score * coverage) + SIMILARITY_BLEND * similarity
```

with `SIMILARITY_BLEND = 0.75`. A chunk only the lexical arm found keeps the
pure rank-and-coverage score, because no cosine was measured for it and
substituting a neutral stand-in would be inventing a signal.

Rank-only relevance is **ordinal**: rank 1 scores 1.0 whether the chunk answers
the question or merely beat everything else. That is why unanswerable queries
used to come back at scores *above* the weakest correct answer — measured, junk
topped out at 0.476 while correct answers reached down to 0.254. Cosine is the
only bounded, measured [0,1] signal in the pipeline, and it is what turns
"nothing beat this" into "this is close to what was asked".

**0.75 is measured, and 1.0 is not better.** The region 0.65–0.80 is a plateau;
past it the number falls off a cliff:

| blend | 0.0 (shipped) | 0.65 | 0.70 | 0.75 | 0.80 | 0.90 | 1.0 |
|---|---|---|---|---|---|---|---|
| top-1 | 0.695 | 0.719 | 0.726 | 0.721 | 0.721 | 0.653 | **0.537** |
| MRR | 0.740 | 0.772 | 0.777 | 0.774 | 0.770 | 0.731 | **0.648** |

Pure cosine reproduces exactly the failure `_relevance`'s docstring already
described: an exact identifier is the strongest lexical signal and the weakest
embedding one, so the remaining 0.25 of rank-and-coverage is what keeps
`ERR_QUOTA_EXCEEDED` findable. The old code was right that raw cosine must not
be the verdict, and wrong that it should therefore be discarded.

### 2. BM25 column weights: `(text, label, section_title) = (1.0, 2.0, 3.0)`

`bm25(chunks_fts)` took no weights, so a query term landing in a 30-character
section title counted exactly as much as one buried in a 4,000-character body.
A human named that section; it is the more deliberate signal.

### 3. Prefix matching now reaches SQLite

`fts_match_expression` quoted every token, making each an exact phrase — so no
prefix ever reached FTS5. Meanwhile `term_coverage` credited `fail` for a chunk
saying `failover`. **Scoring knew about matches retrieval could not find.**
Tokens of ≥4 characters now carry FTS5's prefix operator, applied outside the
quotes so user text still cannot become an operator. Shorter tokens stay exact:
`"of"*` matches most of a corpus and buys nothing.

Note this does not close the `retry`/`retries` gap — neither a prefix nor the
coverage rule crosses that spelling change. A real stemmer would; it remains
deferred, and the semantic arm is what carries such pairs today.

---

## Result

Both columns are live end-to-end runs through the CLI against the same corpus,
the same key and the same container — the only difference is the three source
files, swapped in and out, ~11.5 minutes per run:

| | shipped | tuned |
|---|---|---|
| top-1 | 0.695 | **0.721** |
| MRR | 0.740 | **0.774** |
| recall@5 | 0.811 | **0.847** |
| found in 12 | 0.826 | **0.884** |
| separation AUC | 0.808 | **0.893** |
| worst answerable top score | 0.2537 | 0.2560 |
| best unanswerable top score | 0.4760 | **0.3410** |

The held-out half moved the same way (top-1 0.667 → 0.719, MRR 0.716 → 0.769,
AUC 0.816 → 0.887), so this is not a fit to the training questions.

By question kind, live, top-1 / MRR:

| kind | n | shipped | tuned |
|---|---|---|---|
| `identifier` | 51 | 0.863 / 0.915 | **0.882 / 0.932** |
| `natural` | 42 | 0.810 / 0.883 | **0.905 / 0.937** |
| `multi_term` | 38 | 0.974 / 0.987 | 0.974 / *0.978* |
| `paraphrase` | 59 | 0.288 / 0.328 | 0.288 / **0.389** |

`multi_term` MRR gives up 0.009 — one question moving one place. `identifier`
*improving* is the load-bearing check: the blend was the change most likely to
break exact-token lookup, and it did not.

**The offline replay predicted both runs exactly** — every aggregate and every
per-kind figure, to three decimals, on both the shipped and the tuned build.
That is what makes the sweep behind these numbers worth anything.

Separation is still **not disjoint** — the best junk score (0.341) sits above the
worst correct answer (0.256), so a single global threshold still cannot separate
them. It is much closer than it was, and the overlap band is now narrow enough
that a caller could use one with a stated error rate.

---

## What is still broken: paraphrase queries

By question kind, end to end:

| kind | n | top-1 | MRR |
|---|---|---|---|
| `multi_term` | 38 | 0.974 | 0.978 |
| `natural` | 42 | 0.905 | 0.937 |
| `identifier` | 51 | 0.882 | 0.932 |
| **`paraphrase`** | **59** | **0.288** | **0.389** |

Paraphrase questions — the concept asked for without the document's distinctive
vocabulary — are answered badly, and they are 31% of the key. This is the single
largest remaining gap, and it is precisely the case the resource index exists to
serve: a fact no section summary mentions, asked for in words the document does
not use.

### It is not a ranking failure — corrected 2026-08-12

This section previously claimed the opposite: *"a ranking failure, not a
retrieval failure — the gold chunk is inside the 120-candidate pool for 57 of
59, and the ranking cannot promote it."* **That was wrong**, and the error was
letting "inside the 120-candidate pool" stand in for "found". A chunk at
semantic rank 90 of a 120-candidate pool has not been retrieved in any useful
sense; it is noise that happens to fall inside the cut.

Measured directly, the semantic arm *on its own* — no fusion, no lexical arm,
no `_relevance` — already scores paraphrase at what the whole pipeline scores:

| | semantic arm alone | full tuned pipeline |
|---|---|---|
| paraphrase MRR | 0.383 | 0.389 |

Everything downstream is already extracting essentially all the encoder
provides. Where the arm actually puts the gold chunk, over 59 questions:

| rank 1 | 2–3 | 4–10 | 11–30 | 31+ |
|---|---|---|---|---|
| 15 | 8 | 15 | 8 | 13 |

For contrast the arm alone scores `identifier` 0.776 and `natural` 0.784, where
the full pipeline reaches 0.932 and 0.937 — *there* the ranking is doing heavy
lifting. On paraphrase there is nothing left for it to lift.

### What was tried against it, and did not work

Every row is measured on this corpus and key, offline against the arms dump
except where noted.

| lever | result |
|---|---|
| `dual_blend` — give the semantic arm its own rank-decay term | AUC 0.905–0.910 held out, but paraphrase MRR 0.389 → 0.390 and recall@5 fell. Not adopted |
| reader over-fetch (`INDEX_LIMIT`) 24 → 36…120 | **worse**: paraphrase 0.389 → 0.386 → 0.350. Admitting more candidates admits more competitors that outscore the gold |
| query-adaptive blend, keyed on max `term_coverage` | flat. The signal separates the kinds cleanly (identifier 1.00, paraphrase 0.60), but the best held-out point moves paraphrase 0.365 → 0.374 — about one question |
| weighting the section-summary window above body windows | flat: summary-only 0.378, body-only 0.381, shipped `max` 0.383; a +0.03 bonus reaches 0.393 and costs top-1 and AUC |
| cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`, depth 24) | **worse** — paraphrase 0.373, overall MRR 0.774 → 0.730 — and 1,051 ms/query on CPU, which is unshippable regardless |

The mechanism behind all of it is visible in a single case. *"How do we keep two
agents from overwriting each other in one workspace"* against a section titled
**Concurrency and Locking** whose summary reads "lock mutating commands per
workspace" scores **cosine 0.144** and lands at rank 213. That was verified
against the deployed weights — the stored vectors are consistent, and
recomputing the pair from scratch reproduces 0.1444. The encoder simply cannot
make the inference from "overwriting each other" to "advisory locks". No
reweighting of a number that wrong can rescue it.

---

## Does the embedder pay for itself? Now it does — and that cuts both ways

Measured by recomputing the entire semantic arm with the bundled hashing
embedder over the same corpus and questions. (`HashingEmbedder` is pure Python,
so the arm can be recomputed in process — no redeploy, no re-embed, no window
where the service under test is not the deployed one.)

| | hashing | MiniLM |
|---|---|---|
| **under the shipped formula** | top-1 0.705 · MRR 0.745 | top-1 0.695 · MRR 0.740 |
| **under the tuned formula** | top-1 0.658 · MRR 0.715 | **top-1 0.721 · MRR 0.774** |

Read the first row first: under the shipped formula the hashing embedder is
*marginally better* than MiniLM. That reproduces the 2026-08-11 finding at 7×
the corpus size and settles it — **the model was buying nothing, because the
ranker was not reading it.** The 1.7 GB of image was genuinely dead weight.

The second row is the point. The same change that makes MiniLM worth having
makes the hashing embedder actively harmful: blending a cosine that does not
measure meaning costs 0.047 top-1 against ignoring it. On paraphrase questions
the gap is the whole story — MRR 0.389 vs 0.245, recall@5 0.542 vs 0.373.

### So the blend is gated, not global

potpie's OSS default ships the hashing embedder and downloads no model, which is
a non-negotiable. A constant tuned for the hosted deployment would have
regressed every OSS install silently. So `EmbedderPort` now declares
`calibrated` — whether its cosine means "close in meaning" rather than merely
"ranks" — `IndexSearchResult` carries it out as `similarity_calibrated`, and
`_relevance` blends only when it is set. An embedder that has not declared
itself is read as uncalibrated, so the conservative branch is also the default.

| deployment | embedder | relevance |
|---|---|---|
| potpie OSS default | `local-hashing-v1` | rank × coverage (unchanged) |
| pie hosted image | `all-MiniLM-L6-v2` | blended at `SIMILARITY_BLEND` |

Both halves are measured, and the lexical changes (BM25 weights, prefix) apply
to both — they are a wash on hashing (top-1 −0.010, recall@5 +0.021, worst junk
score 0.476 → 0.429) and a gain on MiniLM.

**What this made newly worth testing:** the embedder is now load-bearing, so
model choice is a real lever for the first time. That test has since been run —
below.

---

## Which encoder? Measured, 2026-08-12

Six encoders over the same windows, the same collapse-to-best-window, and the
same 202 queries, scoring **the semantic arm alone** — which the section above
establishes is the ceiling any ranking change is working under. No index
rebuild: the `windows` table already holds every window, so a model is screened
by re-embedding those strings in memory.

The MiniLM row is a **gate, not a result** — it has to reproduce the arm numbers
measured off the real dump (identifier 0.776, paraphrase 0.383) or the screen is
measuring its own arithmetic. It does, exactly. *(It did not on the first
attempt: a body window stores a `(start, length)` slice and only `section`
windows carry their own text, so reading `windows.text` directly fed the model
992 empty strings. Identifier collapsed to 0.624 and the gate caught it.)*

| model | dim | MRR | AUC | identifier | natural | multi_term | paraphrase |
|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` *(shipped)* | 384 | 0.649 | 0.874 | 0.776 | 0.784 | 0.745 | 0.383 |
| `bge-small-en-v1.5` | 384 | 0.679 | 0.889 | 0.762 | 0.878 | 0.793 | 0.392 |
| `bge-base-en-v1.5` | 768 | **0.713** | 0.854 | **0.859** | 0.853 | **0.859** | 0.391 |
| **`gte-base`** | 768 | 0.701 | **0.902** | 0.812 | **0.881** | 0.807 | 0.410 |
| `e5-base-v2` | 768 | 0.700 | 0.886 | 0.805 | 0.867 | 0.839 | 0.402 |
| `all-mpnet-base-v2` | 768 | 0.656 | 0.799 | 0.707 | 0.820 | 0.702 | **0.465** |

Two things fall out of this table.

**A stronger encoder is a real, broad win — and it is not a paraphrase fix.**
`bge-base` buys +0.083 identifier and +0.114 multi_term; it buys +0.008
paraphrase. The gap that motivated the search is the one that does not close.

**The best paraphrase model is the worst retrieval model.** `all-mpnet-base-v2`
is trained symmetrically — *sentence A means the same as sentence B* — while
bge/gte/e5 are trained asymmetrically, query against passage, MS MARCO-shaped. A
paraphrase question *is* a symmetry task, which is exactly why the ordering
inverts on that column and only that column.

That looked like a case for running both. It is not:

| variant | MRR | AUC | identifier | paraphrase |
|---|---|---|---|---|
| `gte-base` alone | 0.701 | 0.902 | 0.812 | 0.410 |
| `all-mpnet-base-v2` alone | 0.656 | 0.799 | 0.707 | 0.465 |
| max of the two cosines | 0.701 | 0.902 | 0.812 | 0.410 |
| mean of the two cosines | 0.683 | 0.821 | 0.740 | 0.475 |
| RRF over the two arms | 0.694 | 0.721 | 0.774 | 0.457 |

Taking the max is just `gte-base` wearing a second index, because its cosines
run systematically higher. Averaging buys +0.065 paraphrase for −0.072
identifier and −0.081 AUC. RRF throws away the magnitude that separation depends
on and AUC falls off a cliff, 0.902 → 0.721.

And the ceiling says why. On the 59 paraphrase questions, counting where each
model alone puts the gold chunk first: **both 13, only `gte-base` 4, only
`all-mpnet-base-v2` 9, neither 33.** The two are complementary on 13 questions
and jointly blind on 33. Even an oracle that picked the better model per query
would reach 26 of 59 — the models are not disagreeing about paraphrase so much
as both failing at it.

**Conclusion.** `gte-base` is the recommended swap on the strength of everything
*except* paraphrase: it takes the arm's MRR 0.649 → 0.701 and its AUC 0.874 →
0.902, the best separation of the six, which is the metric this retrieval is
chosen on. Paraphrase moves 0.383 → 0.410 and stays the largest gap in the
corpus. Closing it needs something this architecture does not have — the
failures are inference-shaped, and the one inference-capable option cheap enough
to test, a cross-encoder reranker, measured worse and 1,051 ms/query. Nothing
here should be read as that gap being one constant away.

---

## Reproducing

See `pie/services/context-graph/benchmarks/retrieval/README.md`. The short
version: import the fixture corpus, dump the arms once from inside the service
container, sweep offline, then confirm the winner end to end.
