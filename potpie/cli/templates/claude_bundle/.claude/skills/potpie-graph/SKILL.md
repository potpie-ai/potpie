---
name: "potpie-graph"
version: "7"
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

## 1. Start with the read

No status or contract preamble. `potpie status` is the one health check, and
every read header names the pot (`potpie graph status` repeats the counts).
Without `--pot`, a command resolves the pot from the repo you are in (its
registration), then the active pot (`*` in `potpie pot list`, which a repo
mapping outranks). `graph read` prints `pot=<name> (<id>)`, `resolve` and
`search` the id: when it is not the pot you expect, pass `--pot local:<name>`.
`potpie graph catalog --profile read` (text) lists views and match mode, the
full text catalog adds mutation ops; use it after an unknown-view error or a
rejected op — `--task` is accepted and ignored. Text `potpie graph describe
<subgraph> --view <view>` prints a view's filters when a read is refused for a
missing input.

Once a header has named the pot, pass `--pot local:<name>` (or
`managed:<name>`): a bare name is checked on both origins whenever a managed
host is configured, one round trip per command.

## 2. Read — `resolve` first, then one named view

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
exact hit sat at 0.55–0.65. `search` is the same ranker with no intent over
all nine families, so a phrase from a document ranks below recent timeline
rows unless you add `--include docs`. Two needs `resolve` does not serve:

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
| `infra_topology.service_neighborhood` | `--scope service:…` `--depth` `--direction out|in|both` (any other spelling returns no rows, not an error); `--environment` only with `include_unqualified_environment:true` in the scope | dependency blast-radius, env-qualified |
| `features.feature_context` | optional `--scope anchor_entity_key:repo:…` | what a repo/service does (Feature nodes via `PROVIDES` / `IMPLEMENTED_IN`) |
| `decisions.active_decisions` | `--scope service:…` — decisions anchor on services; a repo scope returns none | active decisions |
| `code_topology.ownership_by_path` | `--scope` | who owns a scope |
| `knowledge.document_context` | `--query` / `--scope` | which ingested document sections cover it; hits carry chunk ids, and a section repeats once per claim about it (same chunk id) |
| `knowledge.document_passages` | `--query` | the top `--limit` chunk ids by chunk *text*, no floor and no text: fetch the first one or two with `resource get` |

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
itself and returns only chunk ids — for a phrase you know is in the document
that no summary surfaced; it always fills `--limit`, so fetch the top one or
two. `SECTION_OF`
holds a document together; `DOCUMENTS` points a document (or one section) at
what it covers — assert it when reference material lands. New documents go
through the per-format `potpie-resource-*` skills and `potpie resource import`;
payloads never enter the graph.

### Query expansion

The local embedder is small; recall depends on the query. Expand the user's
words for `prior_occurrences`, `timeline` and `document_context` — "add retry
to the payments client" → also "timeout, flaky, tenacity, backoff, external
call". Those views rank their pool and return up to `--limit` rows however
weak: `--query-threshold` is honoured only by `preferences_for_scope`, so a
full list is not evidence and an empty one means the scope or window is
empty. Judge each row by its score and text. Never pass `--query` to
`preferences_for_scope`.

## 3. Resolve identity — `graph search-entities`

Read with the obvious key first; the header says `items=0` when it is wrong.
Search on a miss, and **before** asserting against an entity no read has shown
you — untyped, because a wrong `--type` guess returns nothing (there is no
`Adapter` node; a Stripe adapter is `dependency:pypi:stripe` plus a `code:`
asset):

```bash
potpie graph search-entities "payments api" --limit 10
potpie graph search-entities "github issue 881" --source-ref <github-pr-or-issue-ref> --limit 10
```

Reuse the returned `key`. `--type` is the PascalCase entity type (`Service`,
not `service`); the wrong case also returns nothing. Inventing a near-duplicate
key (`service:payments` vs `service:local:payments-api`) fragments the graph
and breaks future reads.

## 4. Write — `record` for one learning, a plan for a batch

One fix, decision, or preference is one call, no JSON file:

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

Everything else — topology, timeline events, features, multi-op batches — is a
**semantic** plan (never raw graph CRUD): `propose` creates a server-held plan,
`commit` applies exactly that `plan_id`.

```bash
potpie graph mutation-template --kind bug-fix
potpie --json graph propose --file mutation.json
potpie --json graph commit mutation-plan:01JY8T5C --verify
```

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
  (a batch). If the learning is useful but uncertain, create a
  `graph inbox add` item instead. If nothing durable was learned, do nothing.

Writes are idempotent by `idempotency_key`, so a nudge-driven capture you've already
made will not duplicate.
