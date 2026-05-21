# Context Engine — Design Review (ontology, query surface, adapter contracts)

> Status: **complete (v3).** All three implementation maps (write path,
> connector/reader adapters, resolve/envelope+benchmarks) folded in. Every
> open question resolved (§4). Ready for your review.
>
> Method: read the five canonical docs + `plan.md`; read the load-bearing
> domain code directly (`ontology.py`, `agent_context_port.py`,
> `intelligence_models.py`, `intelligence_policy.py`, `graph_quality.py`,
> `graph_mutations.py`, `source_references.py`, the four core ports); then
> three parallel deep-reads of the write/reconciliation path, the
> connector/reader adapters, and the resolve path. Every claim cites code.
>
> This is a design review, not a bug list. Opinionated by request. The
> engineering discipline here is real; the problems are architectural, not
> sloppy.

---

## Executive summary (read this first)

The vision is right and the *infrastructure* is genuinely well-built. The
problem is that the three things that make the vision real — **one coherent
graph, cross-source identity, and honest confidence** — are each absent,
faked, or pointed at the wrong place. Concretely:

1. **There are three graphs/surfaces pretending to be one.** Writes land in
   a canonical `(pot_id, entity_key)` graph; the lifecycle/conflict/
   supersede intelligence runs on Graphiti's *separate* episodic subgraph;
   and reads come back through three divergent envelope shapes depending on
   `goal`. "One pot, one graph, one story" is the vision; the code has
   three stories.
2. **Cross-source identity — the entire reason the engine exists — is
   unsolved.** The synonyms table is empty; there is no fuzzy/embedding
   resolution. A Linear ticket, a PR, and a doc about "checkout v2" merge
   only if the LLM emits byte-identical keys. One pot is N disconnected
   source-shaped islands.
3. **Every trust signal an agent would act on is cosmetic.** `confidence`
   is one of three constants off a coverage enum; `quality.status` is
   ≈always `watch` by construction; per-fact confidence is usually `None`
   and never decays. The two honest channels (`fallbacks`, `open_conflicts`)
   are not where the contract points agents.
4. **The deterministic ingestion path was built, tested, and never
   wired.** `propose_plan` has zero production callers; everything is
   LLM-re-derived from raw payloads, and soft-downgrade is the default — so
   the strict 50-type ontology is, at runtime, "accept whatever the LLM
   produced." The benchmark gap (mock 82% vs API 53%) is almost entirely
   this: ingestion reliability/latency, not the resolve machinery (which
   works when fed).
5. **The four-tool surface is honest at the tool level and bloated
   beneath:** ~64 include values of which ~30+ are silent no-ops (accepted,
   produce nothing, *no* `unsupported_include`), a 5th tool in the
   manifest, and the documented trust-escalation modes (`deep`/`verify`)
   unreachable via MCP.

What to keep: the hexagonal layout, the spec-driven ontology *machinery*,
the write *plumbing*, the provenance *shape*, the trust-spine *shape*, and
`ContextReaderPort`. **Decided:** deterministic ingestion is deleted
entirely (D0) — this is an LLM-extraction engine and owns that. What to
rebuild on top of that: one graph, identity, one claim/merge model, a
shrunk ontology, a tightened surface — **in that order**. Detail and
reasoning below; sequenced proposal in §5.

---

## 0. The one thing to take away

There are **two graphs in one Neo4j**, and the engine's whole
intelligence layer — supersession, conflict detection, drift, the temporal
story — runs on the *wrong one*.

- **Graph A — the canonical graph.** Nodes keyed `(pot_id, entity_key)`,
  typed edges (`:OWNS`, `:IMPLEMENTS`, …), written by the LLM reconciliation
  agent via `apply_plan`. This is what the ontology validates, what the
  agent contract describes, what readers are supposed to serve, what your
  use cases ("did this bug exist before / how does this feature really
  work") need. **It has no semantic identity resolution (the synonyms table
  is literally empty), no conflict detection, no supersession except a
  manual agent op, and effectively no confidence.**
- **Graph B — Graphiti's episodic extraction.** `Episodic` nodes and
  `RELATES_TO` edges that Graphiti's own LLM extraction produces from the
  episode body. **All** the lifecycle machinery — `temporal_supersede`,
  `family_conflict_detection`, `classify_modified_edges`,
  `list_open_conflicts` — operates here
  (`adapters/outbound/graphiti/episodic.py:344-367`).

So when the docs say "one pot, one graph, one story" and "every change is a
validated `ReconciliationPlan`," that describes Graph A. When they say
"contradictions supersede, conflicts surface, drift is tracked," that
happens in Graph B. The two are structurally separate (write-path agent,
§4.1/§4.3 below) and the agent-facing trust signals are computed from a
third place again (defaulted source refs, PS4).

**And the read side fragments the same way.** There is no single envelope:
`goal=answer/investigate` returns `bundle_to_agent_envelope` (the documented
shape); `goal=retrieve/neighborhood/timeline/aggregate` returns a
*different* shape via the `ContextReaderRegistry` (its own `RouterFallback`
taxonomy, no `IntelligenceBundle`, no confidence/quality machinery); and
MCP `context_search` hand-builds its *own* minimal dict locally
(`mcp/server.py:103-123`) with `coverage.status = "complete" if rows else
"empty"` and `freshness.status` hardcoded `"unknown"` — bypassing both. The
agent contract claims one unified envelope; an agent actually sees three,
depending on which tool/goal it used. **The vision's coherence depends on
graph, lifecycle, and read surface being one thing; they are six.**

Everything below is, in some form, a consequence of this split plus the
fact that the deterministic ingestion path was designed and then never
connected.

---

## 1. Vision recap (so we're aligned)

One graph per **pot** (whole project: every repo/service/doc/tracker/ops
integration/person/agent). Deterministic connectors + a reconciliation
agent feed it; agents read via four tools. Source-reference-first storage.
Declarative ontology. Single validated write path. Every answer carries
provenance/coverage/freshness/quality so the agent decides whether to
trust it.

The vision doc and anti-goals are the strongest artifacts in the repo.
This review is about the distance between that and what the code can
truthfully deliver — and that distance is large in exactly the dimensions
you flagged: ontology realism, surface minimalism, and honest confidence.

---

## 2. What is genuinely solid (credit where due — this constrains the rebuild)

Do not throw these away; a rebuild should preserve them:

- **The hexagonal skeleton and the spec-driven ontology derivation.**
  `ontology.py:1675-1872` builds every family map, freshness table,
  classifier, predicate-family, and endpoint-inference table from the
  catalog at import. The "single-file edit to add an entity" claim is real.
- **The write *plumbing* is robust and hardened.** Admission → idempotent
  ledger append (dedupe on `scope+source_id`,
  `event_admission.py:66-68`) → per-pot coalesced batch → `FOR UPDATE SKIP
  LOCKED` claim (`batch_repository.py:132-149`) → stale-batch reaper with a
  documented lease invariant → durable execution-log resume → prompt-
  injection data-fencing, request/timeout/apply-count ceilings
  (`pydantic_deep_agent.py:916-940`, `:1219-1257`). This is real
  distributed-systems care.
- **The provenance *shape*** (`graph_mutations.py:10-110`) is correct:
  pot, source event id, episode uuid, source system/kind/ref, event
  occurred/received, valid_from/to, confidence, agent, run id, actor. It
  round-trips (written `canonical_writer.py`, read
  `query_helpers.py:402-439`).
- **The structured trust *spine*** (coverage / freshness / quality / drift
  / typed fallbacks / open_conflicts / recommended_next_actions) is the
  right shape and `assess_graph_quality` (`graph_quality.py:151-269`) is
  real logic tied to ontology TTLs.
- **`ContextReaderPort`** is a clean, near-right contract (PS7).

The problem is never the carpentry. It is what the carpentry is pointed at.

---

## 3. Problem sets

### PS1 — A precision ontology fed only by an LLM, with the deterministic path dead

**Confirmed by both impl-maps.** The vision's "deterministic source
connectors" do not feed the graph. `SourceConnector.propose_plan` is
implemented on GitHub/Linear/Notion and exposed on the registry, but its
**only non-test caller is the registry method calling itself**
(`source_connector_registry.py:280`). The reconciliation agent has no
`propose_plan` tool and no path to consume a connector plan. Both webhook
paths normalize then submit through `INGESTION_KIND_AGENT_RECONCILIATION` —
the LLM re-derives the graph from the raw payload every time. The hundreds
of lines of hand-written, tested GitHub/Linear plan compilers are dead
code (plan.md:124-134 "connector-proposed plans as candidate input" — not
implemented).

Consequence for the ontology: a ~50-entity / ~55-edge schema with strict
`required_properties` and enforced lifecycle enums
(`Service`→`(name,criticality,lifecycle_state)`,
`Migration`→`(name,migration_kind,phase)`, …) is populated **exclusively by
LLM extraction over noisy multi-source text**. And **soft downgrade is the
default** (`CONTEXT_ENGINE_ONTOLOGY_SOFT_FAIL` on,
`CONTEXT_ENGINE_ONTOLOGY_STRICT` off — `reconciliation_flags.py:64-83`):
unknown labels dropped, lifecycle coerced to `"unknown"`, unknown edges
rewritten to `RELATED_TO` with a hardcoded `confidence: 0.3`
(`reconciliation_validation.py:165-372`). So the strict schema's real
runtime behaviour is "accept whatever the LLM produced, quietly degrade the
rest." A schema you can't populate to spec isn't a stricter schema; it's a
looser one wearing a strict error message.

**Why it matters for your goals.** You want the graph to capture the *real*
project. A 50-type schema that is never fed deterministically and is
soft-downgraded by default does not capture a real project precisely — it
captures one LLM's best guess, unevenly, with the precision concentrated in
the validator rather than the data.

**Direction.** Decided (see §5 D0): the deterministic-connector ambition is
**deleted entirely** — this is an LLM-extraction engine and will own that.
That makes the second point below mandatory rather than optional:
1. ~~Reconnect a deterministic path~~ — **decided against; remove all
   deterministic-ingestion code (D0).**
2. **Shrink the ontology to what LLM-only ingestion can populate at
   fidelity** (now required, not a choice — see §5 E1).
   Strawman in §5. The test: can a deterministic compiler *or* a single
   extraction reliably produce this entity *with its identity fields* from
   a real artifact? If not, it's a property or a future, not an entity.

---

### PS2 — Four tools is true; the parameter surface is the sprawl

`CONTEXT_INCLUDE_VALUES` (`agent_context_port.py:29-93`) has **64** include
keys; `agent-contract.md` documents **26**. Plus 11 intents, 12 record
types, 11 recipes, 5 source policies, 4 modes, 6 goals, a 13-field scope,
`artifact_ref`, `budget`, `exclude`, `as_of`, and a three-layer resolution
(`include_keys` on specs → implicit `project_map_family`/`debugging_family`
keys → `INCLUDE_KEY_AGGREGATES`, `ontology.py:1705-1754`).

**It is worse than sprawl — most of it silently does nothing.** The
resolve/envelope map confirms **~30+ of the 64 include values have no leg
in `build_evidence_plan` and no provider method** (e.g. `interfaces`,
`schemas`, `integrations`, `branches`, `commits`, `people`, `roadmap`,
`activities`, `periods`, `conflicts`, `metrics`, `oncall`, `teams`,
`agents`). They are accepted as valid vocabulary, produce nothing, and
**do not even emit an `unsupported_include` fallback** (`FALLBACK_ONLY_
INCLUDES` is empty, `agent_context_port.py:324`). An agent that asks for
`include=["risks","roadmap","conflicts"]` gets an empty result and *no
signal that anything was unsupported* — the worst possible failure mode for
a contract that exists to tell agents what it knows.

**The documented trust-escalation is unreachable through the real
transport.** MCP `context_resolve` only ever sends `strategy=hybrid|auto`,
and `_mode_for_strategy` (`context_graph.py:387-390`) maps `hybrid→balanced`
and everything else→`fast`. So `deep` and `verify` modes — the ones
`agent-contract.md` tells agents to escalate to when "coverage or risk
requires it" — **cannot be reached via MCP at all**. Same for goals:
`investigate/timeline/aggregate/neighborhood` are HTTP-only. The agent is
instructed to escalate down a path its transport doesn't expose.

And `context_port_manifest()` (`agent_context_port.py:398-405`) advertises
a **fifth tool, `context_ingest`** (manifest-only — not an actual MCP
`@mcp.tool()`), directly contradicting "four and only four." The four-tool
rule was honored at the tool level and the sprawl reappeared one level down
as a 64-word vocabulary that is half-phantom, partly unreachable, and
self-contradicting in its own manifest.

**Direction.** Hold four tools. Collapse includes to ~8 orthogonal evidence
families an agent actually reasons in ("what is this / who owns it / what
changed / what broke / what was decided / where's the source"); let
`intent` pick defaults so most calls pass no `include`. Make the 60-key
catalog an internal reader-routing detail, never the agent vocabulary.
Delete `context_ingest` from the manifest (fold raw episodes under
`context_record`). Generate `agent-contract.md`'s catalog from code in CI
(plan.md Phase 3 promised this; it didn't land).

---

### PS3 — Confidence is fiction, and it's the most prominent field

Three independent confirmations that the single number an agent is most
likely to branch on means nothing:

- Envelope `confidence` = `_confidence_for_coverage(coverage.status)` =
  **three constants** {complete 0.82, partial 0.55, empty 0.2}
  (`agent_context_port.py:451-456,549`). Not a function of evidence,
  freshness, verification, or conflicts.
- The graph-level confidence it *could* draw on barely exists: the LLM
  plan schema has **no per-entity / per-edge confidence field**; it's a
  single optional plan-wide scalar, **usually `None`**
  (`graph_mutations.py:72-73` emits `prov_confidence` only if set). The
  only confidence ever written deterministically is the hardcoded `0.3`
  penalty on soft-downgraded edges.
- **No temporal decay anywhere** (write-path grep confirmed): the only
  "decay" in the codebase is causal-graph-distance attenuation in
  retrieval, unrelated. `prov_confidence` is written once, never aged.
  `CausalChainItem.confidence` is a hardcoded `0.7`.

So the ontology's design pillar — "every fact carries `confidence` and
`evidence_strength` so agents can weight deterministic facts above
hypotheses" (`ontology.py:14-21,103-104`) — is **fictional in the data**.
Two-decimal precision on a 3-valued enum actively implies a calibration
that does not exist; a pot full of confidently-wrong extractions returns
`0.82`.

**The cosmetic signals are invisible to your benchmark**, which is why this
has survived. `evaluators/retrieval.py` checks include-usage, source-ref
count, and citation; it never asserts that `confidence` or `quality.status`
are *calibrated*. So mock=82% partly *rewards* cosmetic signals passing
unchecked — the number does not validate the trust plane at all.

**Direction.** Delete the fake scalar. Replace with an honest coarse label
(`high/medium/low/unknown`) computed from signals that are actually real:
`evidence_strength` (deterministic-connector vs LLM-guess — make this real
by reconnecting determinism, PS1), freshness status, verification state,
and open-conflict presence. If you keep a number, it must be derived from
those, not from a coverage bucket. And add one benchmark assertion that a
known-stale/conflicted fixture yields a *non*-trusting signal — otherwise
this regresses silently again.

---

### PS4 — The trust spine is well-shaped but fed defaults, not signal

The structured block (coverage/freshness/quality/drift/fallbacks/
conflicts) is the right design and the part that genuinely serves
"suggestions, not facts, go-verify." But the connector impl-map confirms it
is **structurally pessimistic and largely cosmetic**:

- Every ref in `_collect_source_references` is hardcoded
  `freshness="needs_verification"`, `sync_status="needs_resync"`,
  `verification_state="unverified"` (`context_resolution.py:224-337`);
  `last_verified_at` never set at collection.
- The only overwrite (`_apply_verifications_to_refs`) fires **only** under
  `source_policy=verify` + a matched fetch-capable connector + a successful
  live call. Default policy is `references_only` (registry returns empty,
  `source_connector_registry.py:161`); most recipes never escalate.
- `sync_status` is **dead state** — never updated anywhere — yet
  `assess_freshness` keys "needs_verification" off it
  (`source_references.py:326-332`).
- `context_status`'s resolver capability matrix hardcodes
  `summary/verify/snippets` → `available=False` ("not configured on this
  server"); only `references_only` is real (`context_status.py:86-122`).

Net: in the common path every fact reads `unverified` / stale,
`quality.status` is permanently `watch`, the agent's takeaway is "verify
everything," which is informationally identical to "I told you nothing." A
trust model whose output is a constant is not a trust model.

**Direction.** Make the cheap signals real even under `references_only`:
stamp `last_seen_at` + `access` + `source_system` at ingest (no extra
fetch) so freshness/TTL is genuine; reserve `verified` for a real
round-trip; distinguish "not yet attempted" from "attempted, stale" (today
both → `watch`); retire `sync_status` or actually drive it.

---

### PS5 — Drift / merge / conflict exists, but on Graph B, not the graph that matters

Restating §0 as a problem set because it's the core. The real, default-on,
reasonable machinery — `temporal_supersede` (predicate-family auto-
supersede, newer wins), `family_conflict_detection` (pairwise per
`(family,subject)`, classifies contradiction/supersession/overlap, writes
`QualityIssue` nodes), `list_open_conflicts` — **all runs on Graphiti
episodic `RELATES_TO` edges** (`episodic.py:344-367`), which are a
*separate subgraph* from the canonical `(pot_id, entity_key)` graph the
agent writes and readers serve.

The canonical graph itself has:
- **No semantic entity resolution.** `entity_canonicalization.py:40-52`:
  normalize = lowercase/trim/underscore + an **empty `SYNONYMS` table**.
  Cross-source convergence happens *only* if the LLM emits byte-identical
  keys.
- **No conflict detection, no supersession** except the agent voluntarily
  emitting an `InvalidationOp`.
- **Last-write-wins** at Neo4j (`SET e += $props`,
  `canonical_writer.py:113`) vs first-seen-wins inside a plan — opposite
  policies, no source-awareness.
- **No decay, advisory-only staleness.** `expire_stale_facts` and most of
  `MAINTENANCE_JOB_FAMILIES` are recommendation strings with **no
  implementation** (`graph_quality.py:27-39`).

Only 4 predicate families exist (`owner_binding`, `datastore_binding`,
`deployment_target`, `lifecycle_status`), so even on Graph B the
contradiction logic covers a narrow slice — and it's the wrong slice for
your use cases, which need contradiction/merge across
Decision/Feature/Issue/Fix, not ownership bindings.

**Why it matters.** "Did this bug exist before and was it fixed?" / "how is
this feature supposed to work vs how it was implemented vs open bugs?" are
queries over canonical Feature/Issue/Fix/Decision entities with merged
cross-source identity and contradiction awareness. The canonical graph has
none of that; the subgraph that has some of it isn't the one the agent
contract serves. This is the single highest-leverage gap.

**Direction.** Decide whether the canonical graph or the episodic graph is
*the* graph, and put one claim/merge/conflict model on that one.
Recommended: a small general **claim model** on the canonical graph — every
fact is a claim with `source`, `observed_at`, `evidence_strength`,
`confidence`; corroboration raises confidence; contradiction within a
generalized `(subject, predicate)` supersedes by recency×strength or opens
a conflict; confidence ages toward `unknown` past the family TTL instead of
flipping binary. This single model replaces PS3/PS4/PS5 cosmetics with one
real mechanism, and it is worth more than any ontology breadth.

---

### PS6 — Cross-integration identity is the crux, and it is currently unsolved

The entire premise — a project spans repos, Linear, Jira, docs, QA, people
— reduces to: do a Linear issue, a PR title, and a doc heading about "the
checkout feature" resolve to **one** `Feature` `entity_key`? Confirmed
answer today: **only if the LLM emits the identical normalized string**,
because `SYNONYMS` is empty and there is no fuzzy/embedding resolution in
the canonical layer (`entity_canonicalization.py:40-52`). The ontology has
`ALIASES`/`RENAMED_FROM`/`MERGED_FROM`/`SPLIT_FROM` wildcard edges but
nothing populates or reads them as an identity layer.

So in practice one pot holds N source-shaped islands that converge only by
LLM-string-luck. That is the "every agent rebuilds context from scratch"
problem with extra infrastructure — the exact thing the engine exists to
kill.

**Direction.** Treat identity as a first-class, testable contract before
anything else:
- Identity classes: external-id-anchored (PR/Issue/Commit/Deployment —
  trivial, deterministic); slug+alias-table-anchored (Feature/Service/
  Component/Person — the hard, valuable case); content-hash (free-text
  notes/decisions).
- A deliberate canonicalization step (deterministic where possible,
  LLM-assisted where not) that writes an **inspectable alias table** an
  agent can introspect ("these 3 source refs were merged into this
  Feature, here's why; here's the disagreement").
- Without this, PS1's ontology breadth is moot — more entity types just
  means more disconnected islands.

---

### PS7 — Adapter contracts (your #1 priority): verdict

**`ContextReaderPort`** (`domain/ports/context_reader.py`) — **essentially
right. Keep it.** Three verbs (`family/capability/read`), `ReaderCapability`
(intents, requires_scope, cost, backend, `compat`) is exactly enough for
declarative routing; "missing scope → fallback, reader not invoked" is the
correct honest failure. All 10 readers hit real backends; none is a pure
stub (`release_notes` is a thin smoke-test filter, `pr_diff` is honestly
self-flagged `compat`). Two fixable issues, neither in the port:
1. The registry's router is **not fully declarative** — `_resolve_families`
   hardcodes `"semantic_search"`, `"timeline"`, `"change_history"` and a
   goal map (`context_reader_registry.py:32-35,141-159`), contradicting the
   "adding a reader needs no router edit" claim. Fix: push goal/strategy
   affinity into `ReaderCapability` (`auto_for_goals`,
   `auto_for_strategies`) so the router never names a family.
2. `read()` is sync while the provider layer is fully async, and a *second*
   read abstraction (`IntelligenceProvider`) still backs the `goal=ANSWER`
   path, and a *third* hand-rolled shape backs MCP `context_search`. **Three
   read abstractions producing three envelope shapes for one contract.**
   This is the §0 fragmentation on the read side; collapse to one. Related:
   even within the `IntelligenceProvider` path, "fusion" is type-routed
   bucket assignment with **no cross-leg ranking or relevance scoring**
   (`context_resolution.py:675-709`) — each family is an independent list.
   Your cross-domain use cases ("how does this feature work across docs +
   tickets + PRs + open bugs") are exactly the ones that need cross-family
   relevance fusion and won't get it from independent buckets.

**`SourceConnectorPort`** (`domain/ports/source_connector.py`) — **right in
spirit, but has a dead limb and two leaks.** Bundling
read+resolve+webhook+plan+manifest behind one kind-keyed, capability-
flagged contract is the correct call; `ConnectorScope.scope: dict` is the
right escape hatch; the registry's budget/fallback fan-out is solid. But:
1. **`propose_plan(event, context_graph: object)` is dead and leaky.** The
   `object` type is a confessed cycle-avoidance leak; worse, *every*
   connector does `del context_graph` — the parameter is paper-only — and
   the verb is never invoked anywhere (PS1). This is the single biggest
   contract-integrity defect. **Decided (§5 D0): delete it entirely** (verb,
   the `context_graph` param, all connector compilers, ~1000 lines). The
   connector contract drops to read+webhook only.
2. **`ResolverAuthContext.github_token`** (`source_resolution.py:60-70`) —
   a provider-specific field on a nominally source-agnostic auth context,
   read by nothing (`auth` is `del`'d everywhere; credentials actually live
   in injected connectors/factories). Either make `auth` the real
   credential-routing mechanism for N integrations or drop it from
   `fetch`'s signature.
3. **`list_artifacts` is non-uniform and shadowed.** GitHub implements it;
   Linear stubs `return ()`; Notion needs an unwired fetcher. Worse,
   backfill is actually done via *agent tools*
   (`github_list_pull_requests`, `linear_list_issues`) — a parallel,
   undeclared enumeration surface bypassing the port. Two enumeration
   mechanisms is contract drift; pick one.
4. Optional-verb semantics overload `None` ("ignore this event" vs "not
   implemented"). Make "unsupported" raise (registry already handles
   `NotImplementedError`); reserve `None` for "ignored."

**Per-connector reality:** only **GitHub is production-real** (real
webhook HMAC, real live fetch, real diff-chunking). **Linear** is real for
fetch+webhook but its planner is dead and `list_artifacts` is stubbed.
**Notion** wires no fetcher in any container — it produces **zero graph
data in any deployment**; it's a contract-compiles smoke test.

**Stress test (does a new real source fit?).** **Jira fits** — structurally
a Linear clone. One real friction: `_infer_source_system`
(`context_resolution.py:148-174`) derives source from the URL host's SLD,
so `*.atlassian.net` infers `atlassian`, not `jira` — a hidden coupling
that strains any source whose product name ≠ its domain. **Sentry strains
the contract**: incident/event/release has no clean "fetch a stable
artifact / verify it's merged" analog; the natural fit is `propose_plan`
(incident→`Incident`+edges), **which is dead**. Sentry is precisely the
case that exposes the missing deterministic half.

---

### PS8 — Doc vs code drift (a catalogue; the pattern is the point)

- `agent-contract.md` lists 26 includes; code has 64 (~30 of them
  silent no-ops).
- "Four and only four tools" vs `context_ingest` in the live manifest.
- "Connector-proposed plans as candidate agent input" (plan.md Phase 2) —
  built, never wired.
- `dispatch_due_batches.py` (named in architecture.md flow) — **the file
  does not exist** (only a stale `.pyc`); replaced by
  `context_graph_jobs.handle_process_batch`.
- `expire_stale_facts` + most `MAINTENANCE_JOB_FAMILIES` — advisory strings,
  no implementation.
- Resolver capability matrix advertised; `summary/verify/snippets`
  hardcoded unavailable.
- In-code refs to "Phase 7 ontology" (`graph_mutations.py:137`) and
  `implementation-next-steps.md` (`:17`) — a doc Phase 0 ordered deleted.
- plan.md Phase 3 "catalog generated from registry in CI, never drifts" —
  it isn't; it did.

The pattern: docs assert phase-completeness while the surface kept growing
past them and the hard parts (determinism, identity, confidence,
verification) were deferred behind advisory strings and dead verbs. That
undercuts the docs' claimed role as "source of truth for what this project
is" — which matters because your agents are told to trust
`context_status`/the manifest.

---

## 4. Resolved findings — the empirical anchor

All earlier open questions are answered:

1. **Dual read path is real and load-bearing on both sides, divergently.**
   `goal=answer/investigate` → provider+synthesizer+`bundle_to_agent_
   envelope`; `goal=retrieve/...` → `ContextReaderRegistry` → different
   shape; MCP `context_search` → a third hand-built shape. Not vestigial —
   actively three. (folded into §0, PS7.)
2. **MCP exposes 4 tools; `context_ingest` is manifest-only.** Contract
   says 5, transport has 4. `deep`/`verify`/`investigate`/`timeline`/
   `aggregate` unreachable via MCP. (folded into PS2.)
3. **The benchmark gap is upstream of resolve.** mock≈82% measures "does
   the envelope plumb canned data into the right buckets and phrase it" —
   the data-shaping layer, with all caps enabled and `confidence`/`quality`
   never asserted. API≈53% is dominated by the **ingestion/reconciliation
   axis**: in the sample report 6/7 events never reached terminal state in
   180s (≈1116s ingest), so the graph was empty and retrieval cited 0
   source refs. **The resolve/envelope machinery is not the bottleneck —
   it works when fed. The bottleneck is the LLM reconciliation path's
   throughput and reliability, plus the empty-by-default graph.** This is
   the hard empirical confirmation of PS1/PS5: the deterministic path being
   dead isn't a tidiness issue, it's the measured cause of the 29-point
   gap. It also means causal/cross-domain are worst because the mock
   *stubs causal_chain to `[]`* (`mock.py:470-486`) so they're unscored in
   mock and hardest for the real provider — the weakness is structural, not
   noise.
4. **Graph B's intelligence is largely invisible to the agent.** Conflict/
   supersede live on episodic `RELATES_TO`; `open_conflicts` *does* surface
   on the hybrid provider (real, good), but the canonical answer the agent
   reasons over carries none of the episodic supersession state. (sharpens
   §0.)

The empirical takeaway re-prioritizes §5: the cosmetic trust plane is a
real problem for *agent decision quality*, but the **functional** blocker
to the vision is ingestion reliability + identity + graph emptiness, not
the resolve path. Fix order in §5 reflects this.

---

## 5. Proposal — sequenced, decision-first

**Keep (the rebuild is constrained by these, not a greenfield):** hexagonal
layout; spec-driven ontology *machinery*; the write *plumbing* (admission/
batch/claim/reap/resume); the provenance *shape*; the trust-spine *shape*;
four tools; source-reference-first; `ContextReaderPort` ≈ as-is.

The order matters: each step's right shape depends on the previous
decision. Steps D0–D3 are **decisions for you** (small to write down, large
in consequence); E1–E4 are engineering that follows mechanically once the
decisions are made.

### D0 — DECIDED: delete deterministic ingestion entirely _(do first)_

Decision made by the architect: the deterministic-ingestion ambition is
**removed, not revived**. This engine is an LLM-extraction engine and should
own that. No deterministic-plan code, contract surface, or vision language
survives — it is dead weight that has confused the architecture (it is the
"designed, built, tested, never wired" middle state, and the source of half
the PS7 contract leak and the PS8 doc drift).

**Cleanup scope (no code remains regarding it):**
- Delete `SourceConnectorPort.propose_plan` from the port contract.
- Delete every connector's `propose_plan` and the per-source plan compilers
  (`connectors/github/plan.py`, `connectors/linear/plan.py`, the Notion
  stub, ~1000 lines total) and their isolated tests.
- Delete `SourceConnectorRegistry.propose_plan` (its only self-caller).
- Delete the `context_graph: object` parameter leak that only existed to
  serve `propose_plan` (PS7 leak #1 dissolves with it).
- Strip "deterministic source connectors" / "connector-proposed plans as
  candidate input" from `vision.md` and `plan.md` (Phase 2 language). The
  connector contract becomes: `kind`, `capabilities`, `list_artifacts`,
  `normalize_webhook`, `fetch` — read + webhook only. Connectors never
  produce graph mutations; the LLM reconciliation agent is the only writer.
- Update `architecture.md` so the write path shows exactly one producer.

**Consequence — this makes two later steps mandatory, not optional:**
- E1 (ontology shrink) is now **required**: if the LLM is the *only*
  populator, a 50-type strict schema is unfulfillable by construction.
  LLM-only ingestion can only be trusted against a small, loose schema.
- D3 (claim model) is now **load-bearing for correctness, not polish**:
  with no deterministic signal, `evidence_strength` and corroboration are
  the *only* defense against the LLM being confidently wrong.

The benchmark gap (§4.3, mock 82% vs API 53%) does **not** reverse this —
it says LLM ingestion is slow/unreliable, so the fix is to invest in
extraction reliability and a forgiving schema (E1) + claim-weighted trust
(D3), not to keep a parallel deterministic path alive.

_Exit:_ `grep -ri "propose_plan\|_plan compiler\|deterministic connector"`
across `app/src/context-engine` and the docs returns nothing; the
connector contract is 5 verbs; `architecture.md` write path has one
producer.

### D1 — Decide: one graph

Pick the graph the agent contract serves. Recommend **canonical is the
product; Graphiti episodic is an internal extraction substrate only.** Move
conflict/supersession/drift onto the canonical graph (E2). _Exit:_
`architecture.md` has one diagram; no agent-visible signal is computed from
episodic `RELATES_TO`; "two graphs" is gone from this review's §0 because
it's gone from the code.

### D2 — Decide: the identity contract (PS6, the crux)

Write down the identity classes and the canonicalization policy:
external-id-anchored (PR/Issue/Commit/Deploy — deterministic);
slug+alias-table-anchored (Feature/Service/Component/Person — the valuable
hard case); content-hash (free-text notes/decisions). Cross-source
convergence happens **only** through an explicit, inspectable alias table
written by a deliberate canonicalization step. _Exit:_ a test where a
Linear ticket, a PR title, and a doc heading for the same feature resolve
to one `entity_key`, and an agent can introspect *why* they merged.

### D3 — Decide: the claim/merge/confidence model (PS3+PS4+PS5 collapse
into one mechanism)

Every fact is a claim: `{subject, predicate, object, source,
evidence_strength, observed_at, confidence}`. Corroborating claims raise
confidence; contradicting claims within a generalized `(subject,predicate)`
either supersede (recency×strength) or open a conflict; confidence **ages
toward `unknown`** as `now - observed_at` crosses the family TTL instead of
flipping binary. This single model *is* honest confidence (replaces the 3
constants), *is* the real freshness (replaces hardcoded
`needs_verification`), and generalizes conflict beyond the 4 predicate
families. _Exit:_ envelope `confidence` is `high/medium/low/unknown`
derived from claim state; a stale/conflicted fixture yields a non-trusting
signal and a benchmark assertion guards it.

### E1 — Shrink the ontology to fit D0's path (PS1)

Strawman ~12 — `Pot, Repo, Service, Feature, Person` (scope/identity);
`Change` (PR/commit/deploy as one activity with a `kind`), `Ticket`,
`Decision`, `Document`, `Incident`, `Note` (durable record memory),
`SourceRef`. Identity fields required; **everything else optional**.
Promote an entity only when D0's path produces it at >80% fidelity with its
identity. Keep the derivation machinery; just shrink the catalog and relax
required-props. _Exit:_ soft-downgrade rate on a real corpus drops near
zero because the schema now matches what ingestion actually produces.

### E2 — Move lifecycle onto the canonical graph (follows D1+D3)

Re-point `temporal_supersede`/`family_conflict_detection` (or their D3
replacement) at canonical `(pot_id, entity_key)` edges; delete the episodic
duplicate. Implement *one* maintenance job that actually enacts staleness
(today `expire_stale_facts` is an advisory string). _Exit:_ `grep` shows no
lifecycle logic on episodic edges; one stale fact, one test, it actually
expires.

### E3 — Collapse the read surface to one envelope (PS7)

One read abstraction, one envelope shape for all goals and for MCP
`context_search`. Add cross-family relevance fusion (the cross-domain use
cases need it). Drop the 3× redundancy (`answer`/`facts`/`bundle`).
_Exit:_ one envelope builder; `context_search` and `context_resolve` return
the same shape; payload size roughly thirds.

### E4 — Tighten the agent surface (PS2)

~8 orthogonal includes that map to evidence families an agent reasons in;
`intent` picks defaults so most calls pass none; delete the ~30 phantom
includes (or make them emit `unsupported_include` — never silent);
unify MCP↔HTTP so `deep`/`verify` are reachable or removed; delete
`context_ingest` from the manifest; generate `agent-contract.md` from code
in CI. _Exit:_ the contract doc is generated; no include silently no-ops;
the manifest matches the transport.

### Sequencing rationale

D0 first because it is now decided and is pure deletion — removing the dead
deterministic path clears the confusion that the ontology (E1) and the
connector contract (PS7) shape were entangled with. D1–D3 are cheap to
decide and everything downstream forks on them. E1–E4 are then mechanical.
**Identity (D2) and the claim model (D3) are worth more than ontology
breadth or surface polish — they are the difference between "a project
graph" and "N source-shaped islands with a strict schema." With D0
removing all deterministic signal, D3 is no longer polish: it is the only
mechanism that keeps an LLM-only graph honest.**

---

_End of review. The document is structured so each problem set stands
alone for discussion; §5 is the proposed order of attack. Happy to expand
any PS into a detailed design, or to turn §5 into a phased plan.md-style
execution doc once you've made the D0–D3 calls._
