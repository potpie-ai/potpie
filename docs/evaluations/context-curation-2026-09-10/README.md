# Potpie as a context curation surface

Review of `631cf218f6eb18fcaa7394fc1c07ccded7ac16b4`, 2026-09-10.

**Potpie has the right basic architecture, but ingestion and retrieval do not yet preserve the same contract.** Several useful details are stored where the normal readers never use them; some qualifiers disappear; and source refresh can leave a live claim pointing at changed or deleted evidence. Improving ranking cannot recover those losses.

The product promise should be: **“Here are useful leads, why they may matter, where they apply, and the references you can follow.”** An agent should investigate and reason from those leads. Potpie need not synthesize an answer or guarantee that every query is answerable. A weak but honestly labeled lead can be useful. A missing scope qualifier, misleading verification signal, or broken citation is a different problem: it makes the lead unsafe to interpret.

This revises the emphasis of the [earlier retrieval evaluation](../cli-usage-2026-09-10/experiments/README.md): prioritize preservation, applicability, and source navigation before answer-oriented ranking or abstention.

## Evidence and scope

Reviewed all 11 agent-bundle skills, their 10 Claude-plugin copies, both internal reconciliation skills, the repository-ingestion playbook, and agent instruction templates. Traced the CLI into record lowering, resource storage/indexing, all 10 reader families, result projection, and nudge rendering. The [skill inventory](skill-inventory.json) records copy differences.

Ran **35 commands** against a new isolated local FalkorDB Lite pot, `local:curation-review-final-20260910` (`pot_17857ebf79e6`), using the previously built wheels for this revision. The only nonzero exit in that final run is the intentional fetch of a removed test chunk. [Transcript](probes-final.jsonl.gz), [14 checked observations](results.json), [replay script](probe.py), [evidence checker](check_evidence.py).

**132 existing focused tests passed:** [86 CLI](cli-tests.txt), [38 core](core-tests.txt), [8 scope](engine-tests.txt). Their passing alongside the probes demonstrates missing coverage of the complete curation journey; it does not clear the findings below.

These are synthetic contract probes, not an independent agent benchmark. The original exploratory [26-command run](probes.jsonl.gz) and [expanded attempt](probes-expanded.jsonl.gz) are retained. The expanded attempt correctly rejected my wrong `APIContract` key prefix; the final script uses `api_contract:`. A first combined pytest invocation hit a test-package import collision ([log](existing-tests.txt)); separate package invocations passed. No managed backend, live hosted ingestion, large corpus, PDF/OCR extraction, or workbook calculation was exercised. Product code and installed user skills were not changed. Three new scratch pots were used; removal probes affected only their synthetic document.

## Required findings, ordered by impact

### 1. P1 — Source refresh changes the evidence behind an active claim

The probe imported “wait 47 minutes,” recorded a derived claim citing its chunk, then refreshed the document to “wait 5 minutes.” The **same resource URI** returned revision 2's new text while the 47-minute claim remained active. Removing the document left both the derived claim and a document-level `DOCUMENTS` link active; fetching their evidence failed.

The URI contains document/section/sequence but no revision. Refresh replaces the current files. Cleanup searches claims whose **endpoints** are removed section keys; it does not find arbitrary claims whose `source_refs` cite chunks, and it does not include the document endpoint itself. Changed sections are not reconciled as changed evidence dependencies.

Locations: [resource facade](../../../potpie/context-engine/src/potpie_context_engine/application/services/resource_facade.py), `import_dir`, `delete`, `_dependent_claims`; [local store](../../../potpie/context-engine/src/potpie_context_engine/adapters/outbound/resources/local_resource_store.py), `build_import_manifest`; [resource lowering](../../../potpie/context-core/src/potpie_context_core/resource_to_semantic.py).

**Fix:** separate stable document identity from immutable evidence identity. Bind evidence to a revision/content hash and locator; retain the cited snapshot, or explicitly report it unavailable. Index reverse evidence dependencies. On refresh, mark affected conclusions `needs_review` rather than silently treating them as current or assuming they are false. On deletion, report unavailable evidence and handle document-level links as well as section links. Test changed text, removed sections, reordered chunks, and a claim with multiple independent sources.

### 2. P1 — Failed verification strengthens the apparent fix

After recording verification with `outcome=didnt_work`, searching prior bugs returned the fix under `RESOLVED`, with `verification_count: 1`, `is_attempted_failed_fix: false`, and increased corroboration. The failure outcome and its reference were absent. Before that, the full named debug read also omitted the stored `verification_status=unverified`, fix steps, and root cause.

[Record lowering](../../../potpie/context-core/src/potpie_context_core/record_to_semantic.py) puts outcome on the Activity node. [PriorBugsReader](../../../potpie/context-engine/src/potpie_context_engine/application/readers/prior_bugs.py) counts every `VERIFIED` relationship, then removes those rows from returned candidates. Its payload omits the structured claim properties. Thus the named view advertises inline verification relations it never receives.

**Fix:** make outcome a first-class verification fact with time, scope, method, and evidence. Distinguish successful, failed, partial, and unknown checks. Count only applicable successful checks as positive corroboration; return failures as useful counterevidence. Carry fix status and steps in the lead. Keep attempts visible without presenting an unverified attempt as a confirmed resolution.

### 3. P1 — Quick writes do not preserve canonical scope

Two direct reproductions:

- `record ... --scope repo:github.com/curation/demo` wrote to `repo:github-com-curation-demo`; reading decisions with the original repo scope returned zero items.
- A preference recorded with `service:curation-api,path:src/payments` returned for `path:src/analytics`. The stored scope had lost `path`.

[`_scope_target`](../../../potpie/context-core/src/potpie_context_core/record_to_semantic.py) slugifies repo values and can re-key already canonical keys. `_code_scope` and [semantic lowering's `_code_scope_for`](../../../potpie/context-core/src/potpie_context_core/semantic_mutation_lowering.py) retain `file_path`, but omit the advertised `path` alias. No scope creates a synthetic repo anchor based on pot ID rather than an explicit project-wide scope.

**Fix:** share one scope parser/normalizer across reads, records, plans, and source registration. Preserve canonical keys; normalize path aliases before persistence; report the resolved anchor and applicability in the write receipt. Separate identity, applicability, and retrieval scope. Validate unsupported dimensions instead of silently dropping them. Make “unknown scope” distinct from “applies everywhere.”

### 4. P1 — Structured knowledge lands outside the retrieval contract

`record --type decision` requires rationale, but a full `active_decisions` response omitted it. A unique token in that rationale had very weak retrieval similarity and was absent from the returned evidence. The rationale was present on the node in the full raw neighborhood. Likewise, the fix's root cause and steps were stored but missing from the full debug read.

The [record bridge](../../../potpie/context-core/src/potpie_context_core/record_to_semantic.py) builds the decision claim description from summary/title, and the fix claim from summary/symptom. The [lowerer](../../../potpie/context-core/src/potpie_context_core/semantic_mutation_lowering.py) copies selected fields to claims but excludes rationale/root cause. The [inline projection](../../../potpie/context-engine/src/potpie_context_engine/application/services/graph_service.py), `_assemble_inline_relation_items`, reads node properties but retains only display fields. Even `--detail full` cannot recover the omitted fields.

**Fix:** define a typed retrieval card for each family and derive its searchable text from meaningful structured fields. A decision card needs choice, rationale, alternatives, affected scope, and source. A debug card needs symptom, attempt/fix, root cause or uncertainty, checks, and source. Preserve these fields in the result instead of requiring the agent to discover them through an operator view. Avoid forcing agents to duplicate every important detail manually into the summary.

### 5. P1 — The recommended quick write cannot attach proper provenance

The CLI exposes `--detail` and `--scope`, but not the `source_refs` or idempotency fields supported by [RecordRequest](../../../potpie/context-core/src/potpie_context_core/ports/agent_context.py). A plausible `--detail source_ref=https://...` was accepted and discarded. The read cited a generated `context_record:...` ID instead of that source. The CLI receipt also drops metadata containing the mutation handle and claim keys.

This conflicts with the skills' advice to use `record` as the normal one-learning write while preserving evidence. Decision truth is automatically `user_decision`; callers cannot distinguish a user's choice from an agent's inference through this command. Missing confidence defaults to 1.0 downstream, although truth/evidence strength are separate signals.

**Fix:** expose repeatable source references, actor/authority and appropriate truth controls, a stable identity or update mechanism, and a separate searchable description when needed. Reject unknown detail keys. Return canonical entity and claim keys, resolved scope, warnings, and a reopenable receipt. Keep a lightweight path for an honest agent observation without fabricating an external citation or user attribution.

Location: [CLI `record`](../../../potpie/cli/commands/query.py), [record lowering](../../../potpie/context-core/src/potpie_context_core/record_to_semantic.py).

### 6. P1 — A correctly ingested API edge is invisible in its advertised view

A verified `Service EXPOSES APIContract` claim appeared in `graph neighborhood`. The named `infra_topology.service_neighborhood` returned no items, even though its own `inline_relations` advertised `EXPOSES`.

The [view definition](../../../potpie/context-core/src/potpie_context_core/graph_views.py) includes it; the [infra reader's predicate list](../../../potpie/context-engine/src/potpie_context_engine/application/readers/infra_topology.py) does not.

**Fix:** derive supported predicate selection and advertised relations from one family contract. Require every template-emitted relation to have a demonstrated read path. This is the clearest example of good ingestion being defeated by retrieval plumbing.

### 7. P2 — Default text output removes what makes a lead useful

The successful exact-token resource search in human mode returned a chunk ID, generic label, and score. JSON contained the useful excerpt and fetch command; default text omitted both. Claim rows similarly omit source references and explicit environment metadata unless the author happens to include them in prose.

The previous fix improved named `document_passages` output, but [`query.py::_item_body`](../../../potpie/cli/commands/query.py) remains a separate renderer. All skills recommend text for agent reads, so this is the common path.

**Fix:** use the same compact lead projection across `resolve`, `search`, and named views: short hint/snippet, scope/environment, evidence status, source locator, and a next read. Put the effective pot into generated commands. Preserve those essentials even when limiting prose. A generic label plus a high score does not help an agent decide what to fetch.

### 8. P2 — `resolve` lacks the scope controls and behavior its name/help suggest

CLI `resolve` and `search` do not expose the scope fields supported by their request models. A task mentioning production, a path, or a date does not become a structured filter. Intent keywords choose families; debugging defaults omit runbooks, and bare search's “broad” defaults omit prior bugs, preferences, owners, and features. `--mode fast/deep/verify` returned the same candidates and reader bounds; code confirms mode is metadata in this path, not a depth/verification control.

Mandatory preferences also run through task similarity when included in resolve; the scope-only preferences skill exists to work around this. A constraint should not have to resemble the current task to apply. Separately, scope-only preference retrieval fetches a bounded pool before applying scope in Python, so a large pot needs a saturation test.

Locations: [CLI query surface](../../../potpie/cli/commands/query.py), [intent defaults](../../../potpie/context-core/src/potpie_context_core/agent_context_port.py), [orchestrator](../../../potpie/context-engine/src/potpie_context_engine/application/services/read_orchestrator.py), [preferences reader](../../../potpie/context-engine/src/potpie_context_engine/application/readers/coding_preferences.py).

**Fix:** expose explicit scope/environment/time where supported and echo the effective query. Retrieve applicable constraints independently from ranked leads. Keep bounded lead diversity across families; explain excluded families and offer targeted follow-ups. Implement the documented modes or label them unsupported/reserved. A `verify` spelling must not imply source verification that did not occur.

### 9. P2 — Document linkage does not consistently connect scoped retrieval to sections

Linking the **document** to a service returned its `DOCUMENTS` claim, but did not expand its section summaries. This probe's link cited a chunk, so it still offered a valid lead; a link grounded only in an external URL would not automatically carry the imported sections. [`DocsReader`](../../../potpie/context-engine/src/potpie_context_engine/application/readers/docs.py) expands section-level matches, not document-level matches into their sections. [`ResourcesReader`](../../../potpie/context-engine/src/potpie_context_engine/application/readers/resources.py) searches by query/document only, without a graph-based service/repo scope join.

**Fix:** let `DOCUMENTS` establish applicability, then traverse document → sections → evidence within explicit bounds. Reuse that relation to constrain passage retrieval when requested. Group section links and structural claims into one lead rather than spending multiple result slots on the same chunk. Report broader fallback explicitly when no scoped material exists.

### 10. P2 — Ingestion success reports persistence more clearly than curation readiness

Static inspection: the [resource facade](../../../potpie/context-engine/src/potpie_context_engine/application/services/resource_facade.py) writes bytes, indexes them, then writes graph structure. There is no transaction across all three. The skill claim “a failure leaves any prior revision untouched” applies only to the store's atomic replacement boundary, not the entire command. The claim that graph rejection means search finds nothing is also wrong for indexed passage search. [`_import_payload`](../../../potpie/cli/commands/resource.py) returns top-level `ok: true` even when a graph/index component is degraded.

`graph commit --verify` usefully checks readback and quality; it does not check that rationale, qualifiers, citations, or the promised view survived. A section's claimed content hash is accepted from the extractor rather than independently establishing that the stored text is unchanged. Empty sections and more than five chunks warn, despite skills presenting the section range under enforced limits.

**Fix:** report separate, explicit states for stored bytes, indexed text, graph links, summaries, evidence freshness, and retrieval checks. For refresh, expose the affected claims and resumable repair work. Derive/check hashes from stored text. A successful persistence step should not masquerade as complete curation. Add an end-to-end retrieval gate to ingestion, not only a readback gate.

### 11. P2 — Skills sometimes override their own evidence discipline

The graph skill tells agents to treat injected context as “graph truth”; nudge formatting omits truth class and detailed verification state. Ranking for scope does not turn an observation into an instruction or an established fact.

The internal mutation cookbook, internal repo playbook, and public CLI skills also teach incompatible repository/activity identity patterns. The internal agent still loads these documents; this is not merely an archived design. Different APIs can remain different, but their entity identities and evidence semantics must converge when they populate the same pot. The internal prompt references an `entity-resolution` skill that is not in its packaged two-skill directory.

**Fix:** teach “use as attributed context; inspect source and qualifiers before acting.” Generate shared identities, supported commands, and field contracts from the same definitions. Keep internal event control separate from public curation procedures. Verify the internal backfill → public read journey before claiming that both ingestion paths produce equivalent memory.

## Expected retrieval contract, command by command

| Surface | Agent expectation | Required response / current review conclusion |
|---|---|---|
| `status`, `graph status`, `doctor` | Can I use the intended pot? | Effective pot/origin, backend/index readiness, curated inventory, open issues. Counts are health signals, not knowledge completeness. |
| `resolve <task>` | Orient me: applicable constraints and a short set of promising leads. | Echo scope and family selection; separate constraints from evidence; preserve references. Current scope/mode/routing gaps are above. |
| `search <phrase>` | Find a known term, error, entity, or passage to follow. | Exact matches plus semantic leads; explain match kind and omissions. “Broad” should mean all supported relevant families or state its narrower contract. |
| `graph read ... preferences_for_scope` | Which rules apply here? | Prescription, applicability, strength, authority, source, exceptions/conflicts. Scope before ranking; no mandatory-rule loss to task similarity. |
| `... prior_occurrences` | Have we seen this symptom, and what should I inspect or avoid retrying? | Bug, fix/attempt, failed attempts, root cause, version/environment, verification outcomes, source links. Current result drops several of these. |
| `... service_neighborhood` | What connects to this service within these bounds? | Complete supported edges within reported depth/direction/environment/budget; source per edge. Fix `EXPOSES`; show truncation and treatment of unqualified edges. |
| `... feature_context` | What capability exists and where is it implemented? | `PROVIDES` plus `IMPLEMENTED_IN` leads, description, scope, source file/module. Its focused predicate selection is appropriate. |
| `... active_decisions` | What was chosen, why, and what does it affect? | Rationale/alternatives, active/superseded state, affected entities, source and time. Returning only the title is insufficient. |
| `... ownership_by_path` | Who owns the relevant code/service? | Owner, scope, evidence and specificity. The reader is primarily service/repo ownership; it does not resolve a path through CODEOWNERS or a code hierarchy. Rename or implement the advertised path behavior. |
| `... timeline`, `timeline recent` | What changed in this scope/time window? | Source event time, activity, touched targets, references, boundedness. Correlation is a lead, not causation. Query filtering and time ordering should be explicit. |
| `... document_context` | Which curated docs/sections cover this? | Grouped document/section cards, coverage links and chunk locators. Scope traversal needs the document-level expansion above. |
| `... document_passages` | Which source passages match? | Snippet, source title/location/revision, chunk ID, match kind, scoped fetch. It is useful even without summaries; semantic thresholds must not imply truth. |
| `graph search-entities` | Resolve identity before linking or traversing. | Canonical key, type, aliases/display name, scoped supporting refs. Validate entity types; wrong case must not mean “no entity exists.” |
| `graph neighborhood`, `graph inspect` | Expand a known identity across families. | Nodes/claims, qualifiers, sources and traversal bounds. Good escape hatch; ordinary family reads should not require it to recover their own required fields. |
| `graph catalog`, `graph describe` | Discover the real supported contract. | Accurate inputs/predicates, examples and required fields. Derive readers/templates from it; expose unsupported knobs honestly. |
| `resource get`, `resource list --doc` | Fetch evidence / inspect a known document. | Verbatim text, locator, source and revision, neighbors. Add pinned revision access and discoverable document inventory. |
| `graph history`, quality reports | Audit changes or find maintenance work. | Mutation/claim history and actionable findings; distinguish graph age from source freshness. They cannot substitute for evidence availability checks. |
| `graph nudge` | Cheap, bounded contextual reminders. | Same evidence qualifications as explicit reads, plus deduplication. Never elevate retrieved source content into unconditional instructions. |
| `ledger query/pull` | Inspect source-event intake. | These are not curated-memory reads. Query currently emits IDs/kinds, pull emits counts; either provide a hydration path or keep them clearly operator-facing. `--filter` is reserved. |
| `graph export/import`, repair, index status/build/rebuild | Backup, diagnostics, or derived-state maintenance. | Explicit operator scope and outcomes. They are not substitutes for source interpretation or normal agent ingestion. |

## Capture the right information in the right place

The graph should hold compact, attributable knowledge and navigation. The resource store should hold source evidence. The index should be rebuildable from those durable inputs. The agent should not have to understand storage internals to preserve a useful fact.

| Information | Durable home | Retrieval use |
|---|---|---|
| Stable identity, aliases, display name | Canonical entity | Reuse keys across source registration, records, plans and integrations. |
| What this entity is | Entity description/summary | Orientation; do not overload this with every changing relationship fact. |
| One statement about an entity/relationship | Claim with typed subject/predicate/object | Return the actual proposition and its qualifiers. |
| Repo/service/path/environment/version applicability | Normalized claim scope and explicit topology qualifiers | Filter before ranking. An absent qualifier means unknown/unqualified, not universal. |
| Rationale, prescription, root cause, fix steps, alternatives | Typed family fields | Display and index the fields the family's command promises. |
| Likely search terms, aliases, distinctive errors | Retrieval text derived from the above, with optional authored search hints | Improve recall without turning speculative synonyms into factual assertions. |
| Source URL/ref, actor, authority, locator, source revision/hash | Evidence reference attached to the claim | Explain provenance and let an agent reopen the exact supporting material. |
| Full document text, logs, tables, excerpts too large for cards | Versioned resource payload | Fetch selected evidence; keep headings, page/line or sheet/range context. |
| What a document/section covers | `DOCUMENTS`, section structure | Connect scoped questions to source evidence. |
| Event time versus observation/ingestion time | Separate temporal fields | Correct timelines and freshness; ingesting an old event does not make it a new event. |
| A verification attempt and its outcome | Verification record/claim with evidence and applicable version | Distinguish a checked failure from a successful fix. |
| Change to a previous conclusion | Explicit supersession/validity/retraction and affected evidence dependencies | Preserve history and flag conclusions awaiting review. |
| Uncertain candidate or unresolved identity | Inbox with source evidence | Make uncertainty reviewable without silently promoting it to canonical policy. |
| Skipped/unavailable sources and ingestion scope | Ingestion receipt/coverage manifest | Let later reads say “not covered” rather than implying “does not exist.” |

For spreadsheets, also capture units, currency, period, sheet/range, aggregation rule, formula/cached-value status, and missing-data assumptions. Derive only durable, checked conclusions. `data_only=True` can yield stale or absent cached formula results; a successful parse is not a calculation check. Keep numeric provenance next to each claim, not just in a prose section summary.

## Expected ingestion contract, step by step

| Step / command | What it establishes | What completion should mean |
|---|---|---|
| Source selection and `source add` | Source identity and routing metadata | The right pot/repo mapping; **not** ingestion completion. Preserve selected scope/window and unavailable lanes. |
| Harness source inspection | Candidate knowledge from actual source material | Read and cite enough to distinguish facts, events, policies, inferences and unknowns. A filename/title alone is usually insufficient. |
| Entity lookup and `mutation-template` | Reused identities and a valid write shape | Templates teach which fields feed which views. Fill applicability and evidence before committing. |
| `record` | One durable learning | Same provenance, scope and typed fields as a plan; return canonical handles and warnings. Current shortcut loses these. |
| `graph propose` | Validation, diff, conflict/risk review | Schema-valid and semantically reviewable; not yet applied and not proof of source truth. |
| `graph commit --verify` | Apply the reviewed plan and read back its claims | Distinguish applied, readback complete, and quality findings. Then verify discoverability for the intended question. |
| `graph bulk apply` | Bounded application of already curated plans | Preserve per-batch receipts/retry state and the same field contract. Volume does not establish curation quality. |
| Resource extraction + `resource import` | Stored source evidence, searchable text, document structure | Report each stage's readiness. Check extraction integrity, labels, locators and revision identity. |
| `DOCUMENTS` and derived claims | Source applicability and reusable conclusions | Scoped discovery can reach the relevant section, and each derived claim can fetch its exact evidence. |
| Refresh / corrections / `resource rm` | Updated source state and dependencies | List impacted conclusions, preserve audit evidence or mark it unavailable, and reconcile links. |
| Inbox processing | Resolution of uncertain candidates | Apply a reviewed mutation or reject with reason; inbox status alone does not create evidence. |
| Final retrieval check | Future-agent usability | Test a natural query, an exact reference/error, a scoped lookup, and an evidence fetch; add a nearby wrong-scope case. Persistence counts alone are insufficient. |

A useful ingestion gate is **capture → locate → inspect → qualify → refresh**, not merely “N nodes and M claims were written.” Do not require the ingesting agent to pre-answer every future question. Require enough identity, structure, terminology, and source context for another agent to continue the investigation.

## Skill-by-skill changes

Shared material should live in one contract reference. Each use-case skill should be short: when to use it, the first command, what to inspect, how to follow a reference, and what to write back. Product commands should enforce the contract; skills should not accumulate compensating advice for silent behavior.

| Skill | Keep | Change |
|---|---|---|
| `potpie-cli` | Pot routing and setup diagnostics | Stop promising end-to-end atomic import; document actual mode behavior and evidence-readiness states. |
| `potpie-graph` | Read first, identity reuse, plans, inbox, explicit truth | Unify lead rendering; remove “graph truth” for nudges; fix full-field expectations and broad-search omissions. |
| `potpie-project-preferences` | Constraints separate from search; explicit policies only | Fix scope capture; move mandatory scope retrieval into the product; define conflict/exception handling and completeness bounds. |
| `potpie-debug-memory` | Symptoms, failed attempts, sources, verify applicability | Outcome-aware verification, root cause/steps in results, environment/version capture, default route to runbooks. |
| `potpie-infra-architecture` | Qualified topology and source-backed links | Fix `EXPOSES`; obsolete claim that invalid direction silently returns nothing; replace blanket “no environment first” with explicit requested scope and visible unqualified-edge policy. |
| `potpie-change-timeline` | Source event time; correlation is not causation | Encode windows explicitly; do not imply natural-language dates become filters; align “query never empties it” with the reader's relative filtering. |
| `potpie-repo-baseline` | Purpose/features/services with an evidence matrix | Include source revision, declared coverage and durable where-to-look references; run every populated family's retrieval path, not only feature/infra smoke reads. |
| `potpie-source-ingestion` | Harness interpretation, uncertainty inbox, deliberate discovery | Add the capture matrix above, typed evidence and refresh dependencies, and mandatory retrieval/fetch checks after persistence. Remove stale wording that section summaries are the only index. |
| `potpie-resource-markdown` | Heading/code boundaries, source locators, coverage links | Correct atomicity/index claims and advisory caps; preserve heading context and versioned citations; distinguish safe excerpting from reconstructing source payloads. |
| `potpie-resource-pdf` | Page provenance, structured sections, extraction script | Same lifecycle fixes; explicitly report extraction coverage and unavailable image/table content so text-only evidence does not imply full-document coverage. |
| `potpie-resource-spreadsheet` | Header repetition, durable derived facts with evidence | Remove “unnamed rows are effectively unindexed”; validate cached values, units and derivations; revisit dependent claims on refresh. |
| Internal `graph-mutation-plan` | Stable identities and source-backed changes | Reconcile key patterns with public contracts/playbooks, correct conflicting edge-direction prose, and preserve claim-level evidence rather than relying on a plan-level evidence bag. |
| Internal `backfill-enumerate-drain` | Bounded enumeration and resume discipline | Persist coverage/skipped remainder explicitly; demonstrate that backfill and later events produce the same identities and public retrieval cards. |
| Internal repo one-shot playbook | Conservative issue/fix distinction and source event time | It contradicts public repository-key guidance and the mutation cookbook. Unify identities; require retrieval cards and public-reader verification for emitted facts. |
| Agent instruction templates | Brief first-read / durable-write orientation | Align with one curation contract. Avoid making every answer a command transcript; return concise useful context with references, and expose full replay details on demand. |

The two resource design principles are sound: extraction scripts should preserve source text, and raw payloads should not flood graph properties. The absolute “never quote chunk text in your reply” rule goes further than that integrity requirement and obstructs evidence-based explanation. Teach bounded, attributed excerpts separately from source import.

## Proposed result shape

Use one small evidence-card model across human and machine output. The example is a **proposed contract**, not a currently accepted JSON payload:

```json
{
  "effective_scope": {"pot": "local:project", "service": "payments-api", "environment": "prod"},
  "constraints": [],
  "leads": [{
    "kind": "runbook_section",
    "key": "docsection:payments-runbook:rollback",
    "hint": "Rollback steps and post-deploy checks for payments-api",
    "match": {"reason": "Exact error text; linked to requested service"},
    "applicability": {"service": "payments-api", "environment": "prod"},
    "evidence_status": "source_observation",
    "source": {"uri": "https://example.invalid/runbook", "revision": "sha256:...", "locator": "Rollback, lines 41-73"},
    "next_read": {"kind": "resource", "pot": "local:project", "reference": "<revision-bound resource reference>"}
  }],
  "coverage": {"searched": ["docs", "resources"], "truncated": false, "source_coverage": "unknown"}
}
```

Keep relevance separate from truth, authority, freshness, and source accessibility. Replace ambiguous header “confidence” with clearly named retrieval/coverage signals. An empty result should identify searched scope and unavailable capabilities; a weak result can remain a useful lead if labeled. A source need not be fetched automatically, but its retrieval method should be understandable and its revision relationship honest.

## Recommended implementation order

1. **Preserve curation:** canonical scopes, typed record fields, provenance, outcome-aware verification, and revision-bound evidence/dependency tracking. These prevent misleading memory from accumulating.
2. **Make the captured knowledge reachable:** predicate/view parity, family projections, document-to-section scope traversal, and applicable constraints outside semantic ranking.
3. **Make the agent path clear:** one compact lead renderer, explicit query scope and bounds, honest mode behavior, and consolidated skills generated/tested against the contract.
4. **Evaluate independent reuse:** have an ingesting agent curate unseen sources, then a separate querying agent use only installed skills and CLI responses. Measure useful-lead recall, source fetch success, qualifier retention, wrong-scope leakage, duplicate leads, commands/tokens to usable evidence, and refresh correctness. Score what the agent can inspect and substantiate, rather than requiring Potpie itself to answer.

The highest-value next change is to make **`record` and a semantic plan produce equally complete, source-linked, correctly scoped retrieval cards**, while addressing citation lifecycle and failed-verification semantics as separate focused changes. Each should include a source → write → named read → fetch regression. Repeating only rank-at-1 tests would miss the failures demonstrated here.

## Reproduce

From the repository root, using an unused local pot name:

```sh
.venv/bin/python docs/evaluations/context-curation-2026-09-10/probe.py \
  --cli .tmp/cli-usage-experiments-20260910/potpie \
  --pot local:curation-review-another-run \
  --output /tmp/potpie-curation-probes.jsonl
.venv/bin/python docs/evaluations/context-curation-2026-09-10/check_evidence.py
```

The replay deliberately refreshes/removes its own synthetic document to inspect evidence lifecycle. It refuses an existing pot or transcript. The checker validates the frozen final transcript and writes `results.json`; its assertions describe current defects, not desired behavior. It should be replaced by desired-behavior regressions as fixes land.
