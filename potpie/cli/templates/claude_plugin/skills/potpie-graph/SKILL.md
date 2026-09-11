---
name: "potpie-graph"
version: "10"
description: "Use when the task can read or write the project-memory graph through the potpie CLI: discover the contract with `graph catalog`, read named views with `graph read`, resolve entity identity with `graph search-entities`, create validated plans with `graph propose`, commit plans with `graph commit --verify`, inspect quality with `graph quality`, or capture uncertain work with `graph inbox`. Also covers writing retrieval-grade descriptions, fetching ingested document chunks with `potpie resource get`, and responding to nudges."
---

# Potpie Graph Workbench

The graph is project memory: preferences, prior bugs and their fixes, infra
topology, decisions, a timeline of changes, and ingested documents. You are the
intelligence that reads it before acting and writes durable learnings after.
Potpie validates, lowers, commits, audits, and ranks. It does **not** scan a
repository or infer rich facts from prose for you.

Text output for reads; `--json` for `propose`, `commit`, `resource import`,
and anything you parse. `graph describe --examples` renders only with `--json`
and shows read commands only: the write payload shape is `graph mutation-template`.

## 1. Select scope without delaying discovery

When health needs checking, run `potpie status` alongside independent reads;
`potpie graph status` repeats the counts. Every read header names the pot.
Use a known explicit pot selector on all calls. If routing is unresolved,
resolve it before scoped retrieval; otherwise check that returned pot IDs agree
before combining results. Once selected, keep the pot fixed through discovery.
Without `--pot`, a command resolves the pot from the repo you are in (its
registration), then the active pot (`*` in `potpie pot list`, which a repo
mapping outranks). `graph read` prints `pot=<name> (<id>)`, `resolve` and
`search` the id: when it is not the pot you expect, pass `--pot local:<name>`.
`potpie graph catalog --profile read` (text) lists views and match mode, the
full catalog exposes entity types, descriptions, canonical identities, predicates,
and allowed endpoints. Before ingestion, inspect `graph catalog --profile full`
and reuse it through the task; `--profile read` is only a read-view index.
Use `--json` when parsing the catalog. `--task` is accepted and ignored. Text `potpie graph describe
<subgraph> --view <view>` prints a view's filters when a read is refused for a
missing input.

Once a header has named the pot, pass `--pot local:<name>` (or
`managed:<name>`): a bare name is checked on both origins whenever a managed
host is configured, one round trip per command.

## 2. Read — one shared discovery pass

Load relevant use-case skills together and share results across them. A newly
loaded skill does not restart discovery. Reuse earlier reads and hook-injected
context when they cover the same task, pot, and scope and are still current.

Run independent reads concurrently using tool-call parallelism:

- `resolve` once for the task's broad context.
- For code work, scope-only `preferences_for_scope --repo current`, with no
  `--query`. It does not depend on resolve; omit only when equivalent current
  scoped preferences are already available.
- For an entity explicitly named in the task whose canonical key is unknown,
  one untyped `search-entities` lookup per needed identity, alongside resolve.
  Skip this branch when no entity is named or its key is already known.

Start with roughly three memory calls, batching additional independent identity
lookups only as needed. Targeted local file discovery can run alongside them.
Use concurrent calls for these short reads; reserve subagents for substantial,
independent source investigations. Inspect every result, including failures.

Then follow the evidence: use returned keys for a neighborhood or named view,
and batch returned chunk IDs into `resource get`. Those follow-ups can run
concurrently once their inputs are known. Use a symptom or timeline read when
coverage is missing or the task requires a full ordered list. Stop expanding
when the task's evidence and applicable constraints are covered. Keep unknown
key → neighborhood and document hit → chunk fetch dependencies sequential.

Broad discovery and phrase follow-up examples:

```bash
potpie resolve "<the task in the user's words>"
potpie resolve "<task>" --include prior_bugs,docs,timeline
potpie search "<known phrase or entity>"
```

`resolve` infers the intent from the task text (*why / stale / failing* →
debugging; *changed / since / recent* → operations; *how do I* → docs) and
returns one bounded envelope of `[family] subject PREDICATE object · fact
(truth, score)` rows, `… +N more (use --json)` when it cut the list.
`--intent` overrides the inference (*who owns* / *what depends on* / *review*
infer `feature`; the families still come back); `--include` names families.
`confidence` in the header is coverage — did each family fill its page, how
relevant is the best hit — not a verdict: a small pot reads `low` with the
right answer on top. A score is one composite per read (similarity first, then
scope, strength, recency): compare rows within a read, never across reads; an
score includes retrieval relevance, not answer probability. Strong lexical
passage matches are protected from family demotion. `search` is a broad lookup;
`--include docs` searches summaries and chunk text, while `--include resources`
searches text alone. Two needs `resolve` does not serve:

| Need | Read |
|---|---|
| preferences for this code | `graph read --subgraph decisions --view preferences_for_scope --repo current` — by scope, **no `--query`**; this view alone applies an absolute 0.7 similarity floor, which a task sentence rarely clears (`resolve` applies it too, so it lists a preference only on a near-verbatim match) |
| the full ordered timeline | `graph read --subgraph recent_changes --view timeline --format table --time-window 30d --limit 50` — window from the question |

Named views:

```bash
potpie graph read --subgraph debugging --view prior_occurrences --query "refund race timeout" --limit 8
potpie graph read --subgraph infra_topology --view service_neighborhood --scope service:payments-api --depth 2 --direction both
potpie graph read --subgraph knowledge --view document_context --query "deploy rollback" --limit 5
potpie graph neighborhood --entity service:payments-api --detail summary --limit 20
```

`graph neighborhood --entity <key>` is the everything-about-one-entity read:
every relation across subgraphs — decisions, preferences, timeline, features,
topology — in one flat list; `--predicate USES` narrows it.

| View | Inputs | Answers |
|---|---|---|
| `decisions.preferences_for_scope` | `--repo current`, or `--scope service:…,path:…`; no `--query` | which preferences apply to this code |
| `debugging.prior_occurrences` | `--query` (symptom), optional `--scope service:…` | "seen this before? what fixed it" (bug + fix/PR inline) |
| `recent_changes.timeline` | `--time-window`, or `--since`/`--until`; optional `--scope` | recent PRs/tickets/activity for the project pot |
| `infra_topology.service_neighborhood` | `--scope service:…` `--depth` `--direction out|in|both` (invalid values are rejected); `--environment` only with `include_unqualified_environment:true` in the scope | dependency blast-radius, env-qualified |
| `features.feature_context` | optional `--scope anchor_entity_key:repo:…` | what a repo/service does (Feature nodes via `PROVIDES` / `IMPLEMENTED_IN`) |
| `decisions.active_decisions` | `--scope service:…` — decisions anchor on services; use the scope the decision is actually linked to, including a repo | active decisions |
| `code_topology.ownership_by_path` | `--scope` | who owns a scope |
| `knowledge.document_context` | `--query` / `--scope` | which ingested document sections cover it; hits carry chunk ids, and a section repeats once per claim about it (same chunk id) |
| `knowledge.document_passages` | `--query` | chunk-text matches with snippets and fetch commands; weak matches may be filtered or warned about |

Scope keys: `repo`, `path`, `file_path`, `service`, `anchor_entity_key`,
`language`, `framework`, `audience`. A repo key is `repo:<host>/<org>/<name>`
as `source add` registered it (`repo:github.com/acme-corp/acme-shop`); prefer
`--repo current` over spelling it. A preference scope hides only rows bound to
a *different* value of the same dimension: `service:inventory-worker` hides a
`service:checkout` rule, while a repo key hides nothing bound by service, so
`items=0` means nothing is bound to that value, not that the key is wrong.
Entity-relation views list both ends of an edge (`items=2` is one preference
plus its anchor). `--environment` is its own flag; the filter
defaults to `qualified_only`, so `--environment prod` alone drops `USES`,
`DEFINED_IN` and `OWNED_BY`. Compact rows cut the fact at about 120
characters; `--detail full` keeps it whole when the tail matters (`resolve`
prints facts whole). Inspect `coverage` and `quality` before relying on
results; `--json --detail full --relations full --format raw` is for exact
machine processing only.

### Ingested documents: find, then fetch

Ingested documents are `Document` / `DocumentSection` nodes whose section
summaries are claims, so `resolve --include docs`, `search`, and
`document_context` land on them. A section hit carries its chunk ids
(`potpie://res/<doc>/<section>/<seq>`); fetch text with one batched call:

```bash
potpie resource get potpie://res/<doc>/<section>/0000 potpie://res/<doc>/<section>/0001 --with-neighbors
```

`potpie resource list --doc <name>` lists a known document's sections, chunk
ids and first lines in one call. `document_passages` matches the chunk text
itself and returns chunk ids with snippets — for a phrase you know is in the document
that no summary surfaced; it applies a relative relevance filter, so fewer than `--limit` hits may
return. Fetch the strongest supporting passages. `SECTION_OF`
holds a document together; `DOCUMENTS` points a document (or one section) at
what it covers — assert it when reference material lands. New documents go
through the per-format `potpie-resource-*` skills and `potpie resource import`;
payloads never enter the graph.

### Query expansion

The local embedder is small; recall depends on the query. Expand the user's
words for `prior_occurrences`, `timeline` and `document_context` — "add retry
to the payments client" → also "timeout, flaky, tenacity, backoff, external
call". These are ranked candidates, not guaranteed answers. Document reads apply
relative filtering; bugs and timeline may retain weak candidates.
`--query-threshold` is supported by preferences and passage reads, but a high
semantic threshold can discard exact identifiers. Judge source text and match
metadata; a full list is not evidence that the question was answered. Never pass `--query` to
`preferences_for_scope`.

## 3. Resolve identity — `graph search-entities`

Reuse a canonical key already returned by a read. When the task names an entity
whose key is unknown, search concurrently with broad discovery, before a scoped
read; do not guess a key and wait for an empty neighborhood. For a direct
entity question, search then read its neighborhood; broad resolve is optional
when it adds no useful context. Also resolve identity **before** asserting
against an entity no read has shown you. Search untyped because a wrong `--type`
guess returns nothing (`Adapter` is a type; a dependency or file may differ):

```bash
potpie graph search-entities "payments api" --limit 10
potpie graph search-entities "github issue 881" --source-ref <github-pr-or-issue-ref> --limit 10
```

Reuse the returned `key`. When candidates are ambiguous, inspect JSON with
`--supporting-claims 2` for summaries and evidence before choosing an anchor.
An empty scoped read can mean missing relations, not a wrong key.
`--type` is the PascalCase entity type (`Service`,
not `service`); the wrong case also returns nothing. Inventing a near-duplicate
key (`service:payments` vs `service:local:payments-api`) fragments the graph
and breaks future reads.

## 4. Write — classify the knowledge, then choose the writer

Before ingestion or a new kind of graph write, inspect the destination's full
ontology and read the selection guidance below:

```bash
potpie --json graph catalog --profile full --pot <pot>
```

Reuse the contract while the destination/version is unchanged. Choose the most
specific supported entity and predicate for each source claim; do not force a
source to populate every type. Templates illustrate payload shapes, not the
limits of the ontology. When a template lacks the relationship you need, compose
it from the catalog's allowed endpoints instead of substituting a preference.

### Ontology selection

Keep three choices separate: entity/relation type (what the claim means), truth
class (how it is known), and write command (how to store it). A fact stated by a
user is not automatically a preference. Split a passage into separate factual,
decision, and policy claims only when the source supports each one.

| Source meaning | Candidate graph representation; verify endpoints in the catalog |
|---|---|
| Current behavior or capability | `Feature`, linked from repo/service by `PROVIDES`, with supported `IMPLEMENTED_IN` links to repo/service/`CodeAsset` |
| Runtime structure, integrations, API or config | `Service`, `Environment`, `DataStore`, `Dependency`, `APIContract`, `Adapter`, `ConfigVariable`, `DeploymentTarget`, `Cluster`; select supported topology predicates |
| Ownership or team membership | `Team`/`Person`, `OWNED_BY`/`MEMBER_OF` |
| An explicit choice and its rationale | `Decision`, `DECIDED`, and supported `AFFECTS` links |
| Explicit reusable guidance about future work | `Preference`/`Policy`, `POLICY_APPLIES_TO`; preserve the prescription and its source |
| Failure, attempted remedy, observed outcome | `BugPattern`, `Fix`, verification `Activity`; distinguish `REPRODUCES`, `RESOLVED`, `ATTEMPTED_FIX_FAILED`, `VERIFIED` |
| Something happened at a source time | Timeline `Activity`, actor/scope links and `Period` as supported; use an event template |
| Source document or runbook | `Document`/`DocumentSection`, resource import for text, `SECTION_OF`/`DOCUMENTS` for structure and coverage; model facts stated inside it separately |

For example, “the worker uses Redis” is a topology fact; “we chose Redis to
reduce latency” is a decision; “all workers must use Redis” is a policy only
when the source actually prescribes it. “The reconciler exports CSV to S3”
describes a capability, not a logging preference merely because it mentions
an audit log. Do not infer a prescription from implementation alone.

`record` is a convenience for supported learning shapes, not the full ontology.
`feature_note`, `service_note`, `workflow`, `runbook_note`, `integration_note`,
`incident_summary`, `investigation`, `diagnostic_signal`, and `doc_reference`
are free-form records: they do not create the corresponding feature, topology,
or event relationships. Use semantic plans for those facts even for one claim.
Use notes for supplemental context and `graph inbox` for unresolved candidates.

One structured fix, decision, or preference can use one call, no JSON file:

```bash
potpie record --type fix --summary "<symptom → fix>" --detail root_cause="<cause>" --detail fix_steps="<step>" --scope service:<name>
potpie record --type decision --summary "<the decision>" --detail rationale="<why>" --scope service:<name>
potpie record --type preference --summary "<prescription>" --detail policy_kind=<kind> --detail prescription="<guidance>" --scope service:<name>
```

`--type` help lists the `--detail` keys each type requires (`bug_pattern`:
`kind`; `preference` / `policy`: `policy_kind`; `decision`: `rationale`;
`verification`: `target_ref`, `outcome`; `fix`: none); a repeated key builds a
list. `--scope` takes an existing key — an unknown one mints a stub. The reply
is a `record_id` and the mutation count. `fix`, `bug_pattern` and `decision`
keys are minted from the whole summary and a `preference` key from its
`prescription`, so keep both short and lead with the distinctive words.

Topology, timeline events, features, and multi-op batches use a
**semantic** plan (never raw graph CRUD): `propose` creates a server-held plan,
`commit` applies exactly that `plan_id`.

```bash
potpie graph mutation-template --kind bug-fix
potpie --json graph propose --file mutation.json
potpie --json graph commit mutation-plan:01JY8T5C --verify
```

`graph mutation-template` is an unscoped, offline schema helper. It accepts
`--pot <origin>:<name>` for uniform command invocation, but ignores the selector
and does not resolve or validate it. Select the actual target on `graph propose`.

`mutation-template` kinds: `repo-baseline`, `feature`, `preference`,
`preference-policy`, `infra-snapshot`, `bug-fix`, `decision`, `timeline-event`,
`timeline-change` — placeholders you fill from sources you actually read;
`propose` validates every op and names a rejected one by index. Three traps
the template does not show:

- Omit `graph_contract_version`; `pot_id` is overridden by the CLI's resolved
  pot, so any placeholder works.
- `review_required` (the correction ops: `patch_entity`, `retract_claim`,
  `end_relation_validity`, `transition_state`, `supersede_claim`,
  `merge_duplicate_entities`) → re-run
  `potpie --json graph propose --file mutation.json --approved-by <user-ref>`,
  then commit; `commit` alone answers `review_required` again.
- `conflict` on commit means another write moved the graph in between: re-run
  `propose` with the same file. `commit --verify` prints the plan id, readback
  and quality status; `graph history --plan <plan_id>` is for later inspection.

Repo/service functionality is first-class: assert
`PROVIDES` (repo/service → `Feature`) and `IMPLEMENTED_IN` (feature → repo/
service/`CodeAsset`), each `Feature` carrying a compact `summary` and a
retrieval-grade `description`.

The payload is always batch-shaped — `pot_id`, `idempotency_key`,
`created_by`, `operations[]` — and each operation carries `op`, `subgraph`,
`subject`, `predicate`, `object`, `truth`, `confidence`, `description`, and
`evidence`; the template prints it filled with placeholders. Use only
operations advertised by `graph catalog` (`upsert_entity`, `link_entities`,
`assert_claim`, `append_event`, the validity/retraction ops and the audited
corrections). Never hard-delete a claim: use validity, retraction,
supersession, or merge operations according to the catalog policy.

## 5. Capture uncertainty — `graph inbox`

Use the inbox when you have evidence that may matter, but you cannot safely pick
the canonical graph update yet. Inbox items are pending work only; they do not
appear in ordinary graph reads as facts.

```bash
potpie --json graph inbox add --summary "Refund retry PR may relate to the prior timeout bug" --evidence github:pr:acme/payments:955 --subgraph debugging
potpie --json graph inbox list --status pending --limit 20
potpie --json graph inbox claim graph-inbox:abc123 --by user:alice
potpie --json graph inbox mark-applied graph-inbox:abc123 --plan mutation-plan:01JY8T5C --mutation mutation-1 --by user:alice
potpie --json graph inbox mark-rejected graph-inbox:abc123 --reason "not enough evidence" --by user:alice
```

Processing an inbox item is normal graph work: read the relevant views, resolve
identity with `search-entities`, record or propose and commit a mutation if
warranted, then mark the inbox item applied or rejected.

## 6. Inspect quality — `graph quality`

Quality reports are read-only. They surface graph maintenance work but never
repair semantic facts directly.

```bash
potpie --json graph quality summary
potpie --json graph quality duplicate-candidates --limit 20
potpie --json graph quality stale-facts --subgraph infra_topology --limit 20
potpie --json graph quality conflicting-claims --limit 20
potpie --json graph quality orphan-entities --limit 20
potpie --json graph quality low-confidence --threshold 0.75 --limit 20
potpie --json graph quality projection-drift --limit 20
```

If a finding changes canonical meaning, repair it through `graph propose` and
`graph commit --verify`. If the evidence is uncertain, create a
`graph inbox add` item instead. Reserve `graph repair` for operator projection
maintenance such as index or summary rebuilds.

### Truth classes

Pick the truth class honestly — it feeds the ranker:

`authoritative_fact` (explicit source of truth) · `source_observation` (observed
source data read by the harness) · `user_decision` (a person decided) ·
`preference` · `agent_claim` (you inferred it; default when unsure) ·
`timeline_event` (something happened) · `quality_finding`. Durable writes need
evidence **or** an explicitly low-authority truth class.

Do not use the graph as a deterministic code scanner. If a repo, PR, ticket, log,
or document should become memory, the harness reads that source, decides what is
worth recording, resolves identity, and writes a semantic mutation.

For GitHub, Linear, Jira, and similar hosted integrations, use the agent's
integration tools/connectors to pull and hydrate source records first. Do not use
Potpie CLI queue ingestion as the graph update path; after reading
the integration data, write durable facts with `potpie record` or
`graph propose` / `graph commit --verify`, or capture uncertainty with
`graph inbox`.

### Retrieval-grade descriptions (the one rule that matters most)

Every entity and claim carries a `description` — a natural-language **retrieval
card** the local embedder indexes. Write it **for search, not display**: include the
**symptoms, synonyms, and scope** a future searcher would type. Validation only
*warns* on a weak description, but a vague card means the fact never resurfaces.
Compare:

- Weak: `"deadlock fix"`
- Strong: `"Concurrent refund + settle deadlocks payments DB under load; seen as 'refund race timeout' and 'payment deadlock on concurrent settle' in prod; fixed by ordering lock acquisition in services/payments/settle.py"`

## 7. Report back — the commands, then a diagram if it earns its place

Every read above is an argument for the answer you give, so put it on the page.
Show the `potpie` commands the answer relies on **verbatim** — the subgraph,
view, scope, query, and limit included. A reader cannot re-run "I checked the
graph", cannot tell that your `--limit 5` is why the list looks short, and
cannot spot that you read `--environment staging` when they meant prod. Reads
that returned nothing get one summary line, not an echo each
(`3 other reads returned no rows: active_decisions, ownership_by_path, timeline`):
silence reads as a confident negative, and a page of empty commands buries the
ones that mattered.

For a write, name the `record_id` or `plan_id` and whether `commit --verify`
passed. That is the handle for `graph history --plan <plan_id>`, and it is the
difference between "recorded" and "recorded and checked".

Then draw the result when the result is a shape:

| The answer is | Draw |
|---|---|
| three or more entities and the edges between them | `flowchart LR` |
| an ordered run of events — deploys, PRs, incidents | `timeline` |
| a symptom moving through attempts to a fix | `flowchart TD` |

Skip the diagram otherwise. One or two entities, a single claim, a yes/no, or a
list of preferences reads faster as a sentence, and a picture that restates one
line costs the reader time instead of saving it.

A diagram is a claim and inherits the same discipline as a mutation: draw only
edges a read or a source supports, label edges with the predicate the graph
actually uses, keep environment qualifiers on the nodes that carry them, and dash
anything you inferred. Never add an edge to make the picture connected — an
invented edge in a diagram is a fact the reader will repeat.

## Responding To Nudges

A Potpie hook may call `graph nudge` and inject its result into your session. The
hook never reasons — you do.

- **`inject_context`** → treat the injected facts as graph truth for this task; they
  were ranked for your current scope, so use them rather than re-fetching.
- **`instruction`** (e.g. "you resolved `<error>` after editing `<files>` — record
  the bug+fix if non-obvious", or "capture durable learnings") → a *prompt to
  decide*, not an auto-write. Decide the truth class, reuse the keys your reads
  returned, write a retrieval-grade `description`, then `potpie record` (one
  fix, decision or preference) or `graph propose` and `graph commit --verify`
  (structured facts or a batch). Apply the ontology selection above first.
  If the learning is useful but uncertain, create a
  `graph inbox add` item instead. If nothing durable was learned, do nothing.

Writes are idempotent by `idempotency_key`, so a nudge-driven capture you've already
made will not duplicate.
