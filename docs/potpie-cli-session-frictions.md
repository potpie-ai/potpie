# Potpie CLI Session Frictions

Date: 2026-09-10

## Context

This report records the Potpie command-line frictions encountered while
answering a project acronym question, retrieving supporting source material,
updating the managed `pms` graph, and verifying the resulting search behavior
on Windows.

The active environment was:

- Windows PowerShell
- `potpie-context-engine 0.1.0`
- managed pot `pms` (`pot_d41f0f0451d74dc1`)
- managed host `10.82.76.146:8090`

## Frictions

### 1. Unicode output was corrupted by the default Windows encoding

Several successful commands rendered characters as the replacement glyph `�`.
This affected both prose and graph output, including the `ü` in
`Prüfung/Montage/Service`.

Example:

```powershell
potpie resolve "What does PMS stand for in this project?" --include docs
```

Impact:

- Evidence was harder to read.
- A user could not reliably distinguish the intended German expansion.
- Successful commands looked partially broken.

Workaround:

```powershell
$env:PYTHONUTF8='1'
```

Potpie should emit UTF-8 reliably on Windows without requiring callers to set a
Python-specific environment variable.

### 2. Unicode output caused commands to crash after partial success

Some commands produced useful output and then exited non-zero because the
console codec could not encode a character:

```text
error: 'charmap' codec can't encode character '\u2060' ...
error: 'charmap' codec can't encode character '\u2192' ...
error: 'charmap' codec can't encode character '\u21b3' ...
```

Affected commands included:

```powershell
potpie resource get <chunk-ids> --with-neighbors
potpie doctor
potpie graph read --subgraph features --view feature_context ...
```

Impact:

- Commands performed their read but reported failure.
- Automation could not trust the exit code.
- Output could be truncated before the relevant evidence.

Workaround:

```powershell
$env:PYTHONUTF8='1'
```

The CLI should configure a UTF-8 output stream or safely replace unsupported
characters instead of crashing during rendering.

### 3. A wrong pot origin produced a misleading daemon error

The active pot was managed, but this command was attempted:

```powershell
potpie graph search-entities "PMS system service software" --pot local:pms --limit 20
```

It failed with:

```text
Cannot reach the local host to resolve 'pms': Potpie daemon is unavailable
```

The correct selector was:

```powershell
--pot managed:pms
```

Impact:

- The error suggested an unhealthy daemon rather than an origin mismatch.
- It prompted unnecessary daemon diagnosis even though `potpie status`
  subsequently reported the managed backend as ready.

Suggested improvement:

- If a pot of the same name exists on another configured origin, mention it:
  `pms was not found locally; managed:pms exists`.

### 4. `graph mutation-template` did not accept `--pot`

Most graph commands accepted explicit pot selection, but this command rejected
it:

```powershell
potpie graph mutation-template --kind feature --pot local:pms
```

Error:

```text
Error: No such option: --pot
```

Impact:

- Pot scoping is inconsistent across neighboring graph commands.
- Scripts cannot apply one uniform invocation pattern.

Although the template is schema-only, accepting and ignoring `--pot`, or clearly
documenting why it is unsupported, would reduce surprise.

### 5. Top-level help failed with an internal error

This command did not show help:

```powershell
potpie --help
```

It returned:

```text
error: Unexpected internal error.
next: re-run with --verbose to see the traceback
```

Impact:

- The primary discovery path failed while troubleshooting.
- Users could not inspect service or daemon management commands.

Top-level help should not initialize optional runtime components that can make
help rendering fail.

### 6. `admin.inspection_slice` accepted an unsupported `--query` option

The generic `graph read` CLI exposed `--query`, so this invocation looked valid:

```powershell
potpie graph read --subgraph admin --view inspection_slice `
  --query "PMS" --pot managed:pms --limit 20 --detail full
```

It returned:

```text
unsupported_filter=query
(no rows)
```

Impact:

- The command exited successfully despite not executing the requested filter.
- The `(no rows)` output could be mistaken for a genuine empty result.

Suggested improvement:

- Reject unsupported filters with a non-zero exit code before running the read.
- Include the supported filters in the error.

### 7. `--source-ref TEXT` was serialized with the wrong type

`potpie graph read --help` documents:

```text
--source-ref TEXT
```

But passing one value to `admin.inspection_slice` failed:

```powershell
potpie graph read --subgraph admin --view inspection_slice `
  --source-ref potpie://res/pms-service-software-architecture-va20a/introduction-and-definitions/0000 `
  --pot managed:pms --limit 50 --detail full
```

Error:

```text
Type mismatch: expected List or Null but was String
```

Impact:

- The documented CLI shape did not match the backend request schema.
- Exact evidence inspection was blocked.

The CLI should wrap a single `--source-ref` value in a list, or expose a
repeatable option whose type matches the backend contract.

### 8. A feature could not anchor `features.feature_context`

This command returned no rows:

```powershell
potpie graph read --subgraph features --view feature_context `
  --scope "anchor_entity_key:feature:pms-service-software" `
  --pot managed:pms --limit 20 --detail full
```

Anchoring the same view through the repository returned the feature data.

Impact:

- Inspecting a known feature required discovering and using an indirect
  repository anchor.
- The input name `anchor_entity_key` appears more general than its effective
  repo-or-service restriction.

Suggested improvement:

- Validate the anchor type and report that only repository or service keys are
  supported.
- Alternatively, support a feature key by returning the feature and its
  implementation/documentation relations.

### 9. Passage retrieval returned weakly related results without a threshold

Queries testing possible acronym expansions returned unrelated passages:

```powershell
potpie graph read --subgraph knowledge --view document_passages `
  --query "Product Maintenance Service PMS" --limit 5
```

The view fills the requested limit even when similarity is weak.

Impact:

- A full result page can look like supporting evidence when it is only a ranked
  fallback.
- Several query variants and source reads were needed to find the explicit
  abbreviation table.

Suggested improvement:

- Show an explicit weak-match warning.
- Support an optional similarity threshold for `document_passages`.

### 10. Correct knowledge existed but was disconnected from the canonical node

The graph already contained:

```text
feature:pms-abbreviation-prufung-montage-service
```

It documented the correct full form, but it was not attached to:

```text
feature:pms-service-software
```

Impact:

- Search could find the isolated abbreviation feature, but the canonical PMS
  node did not carry the definition.
- Generic retrieval did not reliably return the full form.

Resolution in this session:

- Updated `feature:pms-service-software` with the full form.
- Added an authoritative `DOCUMENTS` claim from the architecture definition
  section to the canonical feature.

This is primarily a graph-quality and ingestion-linking issue rather than a CLI
parser defect.

### 11. `graph commit --verify` timed out after the commit succeeded

The validated plan was committed with:

```powershell
potpie --json graph commit mutation-plan:4d4698d13f3a `
  --verify --pot managed:pms
```

After 30 seconds the client exited with code 2:

```text
Managed (10.82.76.146:8090) did not answer within 30s (ReadTimeout).
That is a client-side deadline, not a failure: the request may still be
running there.
```

The command had in fact committed successfully. Confirmation required:

```powershell
potpie --json graph history --plan mutation-plan:4d4698d13f3a `
  --pot managed:pms
```

Impact:

- The most sensitive operation ended in an ambiguous state.
- Blind retrying could duplicate work if idempotency were absent or incorrect.
- Verification required a separate manual recovery flow.

Suggested improvement:

- Poll plan status automatically after a commit timeout.
- Return a distinct `unknown_completion` status and the exact history command.
- Allow a configurable client timeout for managed commits.

### 12. Generic search and focused document search ranked very differently

After the graph update, these generic lookups still prioritized unrelated
high-scoring facts:

```powershell
potpie resolve "What is the full form of PMS?" --include docs --pot managed:pms
potpie search "PMS full form" --pot managed:pms
```

The focused lookup returned the desired definition first:

```powershell
potpie search "PMS full form" --include docs --pot managed:pms
```

Impact:

- Users need to understand family selection to retrieve an elementary project
  definition reliably.
- A newly written authoritative claim can lose to unrelated topology facts in a
  generic search.

Suggested improvement:

- Infer acronym-definition intent from phrases such as `full form`, `stands
  for`, and `abbreviation`.
- Weight exact definition language and authoritative documentation more heavily
  for that intent.
- Consider returning entity metadata matches alongside claims in `resolve`.

## Priority Summary

| Priority | Friction |
| --- | --- |
| High | Windows Unicode crashes and corrupted output |
| High | Ambiguous successful commit reported as timeout failure |
| High | Investigate managed `--source-ref` type error; CLI serialization is not the confirmed cause |
| High | Broken top-level help reproduced through the Unicode failure path |
| Medium | Misleading managed-versus-local pot error |
| Medium | Generic acronym lookup ranking |
| Medium | Unsupported filters accepted by generic `graph read` |
| Medium | `document_passages` ignores the advertised query threshold |
| Low | Inconsistent `--pot` support on schema-only commands |
| Low | Feature anchor behavior is narrower than its name suggests |

## Code Review Findings and Next Steps

Reviewed against Potpie `swid` at `4936b6ce`, including the extension and
Copilot ACP launch code in the sibling `pie` checkout. The session observations
above are preserved; the findings below qualify their proposed causes. The
managed host and its graph were not inspected during this review.

### 1. Fix Windows output encoding first (frictions 1, 2, 5)

Implementation status: the CLI now configures its stdout/stderr text wrappers
for UTF-8 before dispatch, including eager help. Streams are reconfigured in
place, caller-provided in-memory streams are preserved, and malformed Unicode
is escaped rather than crashing output. Five regression cases cover legacy
pipe encodings, human/JSON output, stderr, and caller-owned streams; all 80
encoding, usage, error-contract, and output tests passed on macOS. Native
Windows PowerShell and VS Code Copilot ACP validation remains pending.

Confirmed before the fix: the CLI inherited its output encoding, and help contains `→`.
Forcing `PYTHONIOENCODING=cp1252` reproduced the reported top-level help error;
the verbose traceback was a `UnicodeEncodeError` during help rendering.
UTF-8 help succeeded. Optional runtime initialization is therefore not needed
to explain this failure.

Next steps:

- Completed: configure CLI stdout and stderr for UTF-8 before help or command
  output is rendered, preserving redirected and in-memory text streams.
- In `pie`, establish UTF-8 in the Windows environment passed to the service,
  Copilot, and CLI helpers. The direct extension command runner decodes UTF-8,
  but setting a decoder does not configure the child process's encoder.
- Trace an actual Copilot ACP shell invocation to verify where command bytes
  become text; the exact corruption boundary inside Copilot is still unverified.

Acceptance: `--help`, `doctor`, `resource get --with-neighbors`, and graph reads
complete without encoding errors through Windows PowerShell and the VS Code
Copilot ACP flow. German text and the reported Unicode characters survive
stdout/stderr capture, with no replacement glyphs. Cover JSON output as well.

Code: `potpie/cli/main.py`, `potpie/cli/ui/output.py`; in `pie`,
`apps/vscode/src/host/runCommand.ts` and
`packages/py/orchestrator/src/pie_orchestrator/harness/acp.py`.

### 2. Make commit completion recoverable (friction 11)

Confirmed: commit and verification share one synchronous RPC with the default
30-second deadline. The server persists the committed state before verification.
Same-plan retry and concurrent-commit guards already exist; missing idempotency
is not the confirmed problem.

Next steps:

- Return the durable commit receipt separately from verification, preserving
  distinct commit and verification outcomes in the CLI response.
- On a commit timeout, perform bounded status polling for that exact pot and
  plan. If completion remains unknown, return a machine-readable
  `unknown_completion` outcome and the fully scoped history command.
- Support a configurable managed-operation deadline; increasing the timeout
  alone does not resolve unknown completion.
- Ensure retrying an already committed plan with `--verify` can obtain the
  verification outcome without applying the mutation again.

Acceptance: simulate a successful write followed by verification lasting more
than 30 seconds. The CLI either recovers the receipt or reports unknown
completion with an actionable recovery command. Retrying the same plan does
not duplicate writes, including while the first attempt is still running.

Code: `potpie/daemon/client.py`, `potpie/cli/commands/graph.py`, and
`potpie/context-core/src/potpie_context_core/workbench_service.py`.

Implementation in the working tree (2026-09-10; not yet shipped):

- `graph commit --verify` requests a durable receipt with verification deferred,
  then calls the read-only `verify_commit` RPC. The pre-commit quality counts
  and status are persisted so separate verification retains regression checks.
  Older plans without a baseline report that comparison as unavailable.
- A commit timeout or an active concurrent commit triggers up to three
  `commit_status` reads against the already resolved host, pot ID, and plan ID.
  Polls use a two-second RPC timeout and half-second intervals. Recovery never
  submits another mutation. Durable failures retain their status and guidance.
- Unresolved commits report `unknown_completion`, exit 2, and include
  `potpie --json graph history --plan <plan-id> --pot <origin>:<pot-id>` in human
  and JSON output. A verification timeout retains the successful commit receipt,
  reports `verification.status=unknown_completion`, and supplies a scoped retry.
- `--timeout <seconds>` or `POTPIE_GRAPH_COMMIT_TIMEOUT` configures the commit
  and verification RPC timeouts; the default remains 30 seconds. Poll timeouts
  are independent. This option applies to remote RPCs, not in-process calls.
- Retrying a committed plan with verification runs readback without applying
  the mutation again. The new read methods also have pot-scoped authorization
  entries in sibling `pie/services/context-graph`.

Validation: CLI/RPC regressions cover bounded polling, exact host/pot/plan
scoping, human/JSON recovery guidance, durable failures, timeout configuration,
and separate verification outcomes. Service tests cover persistent JSON-store
compare-and-set transitions, retained quality baselines, concurrent commits,
and retry verification without mutation. Managed authorization tests check
both allowed and cross-tenant access. A real loopback HTTP probe delayed
verification for 31 seconds: the client returned at 30.02 seconds with the
committed receipt and unknown verification, then verified the same plan while
the first verification was still running; exactly one mutation was applied.

Deploy the updated core/service and managed authorization policy with the CLI.
The original Windows/managed session remains to be rerun after deployment.
The separate surface-manifest check currently fails on the pre-existing
managed-only `usage` surface mismatch; this change does not alter that surface.

### 3. Reject unsupported read filters (friction 6)

Implementation status: done. The graph service now returns `ok=False` with
`status=unsupported_filter` when a requested filter is outside the view's
contract. The message names the offending filter and lists the view's
supported filters (or states that the view accepts no filters); the structured
`unsupported` entry keeps `detail.supported_filters`. The CLI renders that
message and exits 1 in human and JSON modes, the daemon UI API returns 400,
and when a required scope is also missing both problems are reported in one
message under the `missing_required_scope` status. Valid reads with no matches
still return `ok=True` with empty items. Regression tests cover the service,
the CLI envelope in both output modes, and the UI router.

Confirmed before the fix: the service detected unsupported filters but used
`ok=not missing`, so an unsupported filter without missing scope returned
`ok=True` and no items. The CLI already exited non-zero when a read result had
`ok=False`.

Next steps: completed as described above.

Acceptance: `admin.inspection_slice --query PMS` exits non-zero in human and
JSON modes, names `query` as unsupported, and lists the supported filters.
Valid reads with no matches continue to succeed with empty results.

Code: `potpie/context-engine/src/potpie_context_engine/application/services/graph_service.py`
and `potpie/cli/commands/graph.py`.

### 4. Honor passage thresholds and expose weak evidence (friction 9)

Implementation status: explicit passage thresholds are now enforced. Omitting
`--query-threshold` preserves relative filtering; supplying a value requires
calibrated semantic similarity and excludes hits with no measured similarity.
Lexical matches cannot bypass an explicit threshold. Uncalibrated or disabled
indexes return an actionable validation error instead of ignoring the option.

Human output shows the threshold mode, match mode, calibration, and warnings
for weak, uncalibrated, or unmeasured evidence. JSON preserves the diagnostics
under `result.coverage[].metadata` and carries warnings in the graph envelope;
JSONL sends warnings to stderr while keeping stdout rows parseable. Generic
search/resolve also expose the reader's warnings.

For example, on a host with calibrated passage similarity:

```powershell
potpie graph read --subgraph knowledge --view document_passages `
  --query "PMS full form" --query-threshold 0.5 --pot managed:pms
```

Review baseline: the flag existed but `ResourcesReader` ignored it. A synthetic
hit with similarity `0.1` survived requested thresholds of `0.0`, `0.7`, and
`0.99`. Regression coverage now checks explicit boundaries, unchanged default
recall, lexical bypass prevention, invalid values, unsupported calibration,
full pages of weak evidence, and diagnostics through CLI/RPC serialization.
Validation: 129 CLI/presentation tests and 158 reader/envelope/graph-contract
tests passed in separate package runs; lint, formatting, and diff checks passed.

Remaining validation: rerun the original acronym queries against the managed
corpus and inspect both relevant definitions and queries with no answer.
Threshold comparisons use measured similarity, not a probability that a
passage supports a proposed acronym expansion.

Acceptance: an explicit threshold changes eligible semantic results as
documented. Test a corpus with no answer, partial acronym-token matches, exact
definition passages, and both calibrated and uncalibrated index modes.

Code: `potpie/cli/commands/graph.py`,
`potpie/context-engine/src/potpie_context_engine/application/readers/resources.py`,
and `potpie/cli/read_presenter.py`.

### 5. Fix source-reference filtering for legacy scalar data (friction 7)

Implementation status: reproduced the exact `Type mismatch: expected List or
Null but was String` error against isolated embedded FalkorDB with a scalar
`source_refs` edge property. The current CLI option is already repeatable,
becomes a tuple, survives RPC with that type, and is converted to a list by
the FalkorDB adapter. The reproduced failure is in the stored-property filter,
not CLI argument wrapping. This does not yet establish the managed host's cause.

Completed:

- Ordinary claim scans and both FalkorDB/Neo4j vector query predicates now
  wrap scalar `source_refs` as a single list element before `any(...)`.
  Native arrays, missing values, and the singular `source_ref` retain their
  behavior. Matching remains exact and repeated filters use OR semantics.
- The shared Cypher edge writer persists a scalar `source_refs` input as a
  native single-element array, preventing that representation on new writes.
- Added real FalkorDB regressions for scalar/array/missing/empty properties,
  single/repeated filters, exact matching, pot isolation, vector filtering
  without lexical fallback, and writer round trips. Both new reader cases
  failed before the fix, including the exact reported type error.

Validation: all 8 embedded FalkorDB integration tests, 64 adapter/writer unit
tests, and 2 CLI source-reference tests passed (74 total). Ruff lint, formatting,
and `git diff --check` passed.

Remaining deployment verification:

- Record the deployed CLI/server revisions, inspect the affected claims'
  stored `source_refs` type, and retry the original managed command after
  deploying the server fix. Capture the traceback if the failure persists.
- Scalar legacy records are now readable without migration. If a storage
  audit finds JSON-encoded arrays or other malformed representations, define
  a targeted migration after confirming their format; this fix does not
  parse JSON array strings inside Cypher or change existing stored records.
- Validate Neo4j against a live deployment; the real database regressions
  currently exercise embedded FalkorDB.

Acceptance: single and repeated source refs retrieve the expected claims on
the managed backend. Local database coverage exercises the failing stored
representation rather than only checking CLI forwarding.

Code: `potpie/context-engine/src/potpie_context_engine/adapters/outbound/graph/`
(`canonical_claim_query.py`, `falkordb_reader.py`, `neo4j_reader.py`, `cypher.py`),
with regression coverage in `tests/integration/test_falkordb_roundtrip.py`.

### 6. Improve definition retrieval and verify graph linking (frictions 10, 12)

Baseline: mixed envelopes weight docs at `0.65` and resources at `0.60`, versus
`1.0` for topology and other project-memory families. The reported acronym
questions fall through intent inference to `feature`.

However, explicit `resolve --include docs` selects only the docs reader in this
checkout. Cross-family weighting cannot explain poor ordering within that
call. Updating entity metadata alone also does not guarantee retrieval through
the claim-based readers.

Investigation and implementation scope:

- Capture JSON results, inferred intent, item families, score breakdowns, and
  deployed revisions for the three reported commands. Compare resolve and
  search with identical query text and includes to isolate routing from wording.
- Add acronym-definition queries to a retrieval evaluation set, including
  incorrect expansions and unrelated topology distractors.
- Recognize definition intent and evaluate intent-dependent family weights
  and exact-definition signals. Check ordinary project-memory queries for
  regressions before changing the global weights.
- Verify the session's canonical-node update and `DOCUMENTS` claim in the
  managed graph. Inspect the ingestion/linking history before attributing the
  disconnected abbreviation node to a specific ingestion defect.

Acceptance: generic and focused lookups rank the explicit definition highly
and return its supporting chunk references. Incorrect expansions do not look
like corroborated answers, and existing project-memory retrieval remains useful.

Code: `potpie/context-core/src/potpie_context_core/agent_context_port.py`,
`potpie/context-engine/src/potpie_context_engine/application/services/envelope_builder.py`,
and `potpie/context-engine/src/potpie_context_engine/application/readers/docs.py`.

Implementation in the working tree (2026-09-10; not yet shipped):

- Added a `definition` intent for named acronym questions (`full form`,
  `stands for`, `abbreviation`, and `acronym` forms). Both agent `resolve` and
  `search` infer it, including when `--include docs` is explicit. An explicit
  intent takes precedence; `--intent definition` also accepts a bare acronym.
  Ordinary search still defaults to `unknown`.
- Definition reads use docs/resources/features/topology, with family weights
  `1.0`/`0.95`/`0.5`/`0.5`. All other intents retain their existing weights.
  Docs recall uses the extracted term, so question words do not dominate its
  bounded candidate pool.
- Docs and passage candidates receive definition-specific ranking before
  truncation. The score combines the original reader score (`0.35`), explicit
  affirmative definition wording (`0.40`), and source-backed claim authority
  (`0.25`). Prose and parenthesized expansions are supported, including
  `PMS (Prüfung/Montage/Service)`. Negated, questioning, and recognized uncertain
  wording receives no affirmative-definition bonus.
- Authority requires a source ref, stored `authoritative_fact` or
  `source_observation` truth, and attested/deterministic evidence. Raw passages
  receive no invented authority. Truth, corroboration, and source refs remain
  separate; competing expansions remain separate evidence. Definition wording
  is a relevance heuristic, not proof that an expansion is correct.
- JSON score breakdowns now expose the original score, definition factors,
  reader score, and family weight. Supporting chunk ids and passage fetch
  references survive ranking.

Validation: the [evaluation fixture](../potpie/context-engine/tests/fixtures/retrieval/definitions.json)
includes a sourced definition, an incorrect expansion, a negated expansion,
a question, another acronym, and topology/document distractors. The
[JSON comparison](evaluations/definition-retrieval-comparison.json) records the
three reported command variants plus two identical-text/include controls.
All five move from a distractor/topology result to the sourced definition at
rank one. This is a synthetic local policy comparison using the previous
explicit intents as its baseline, not a managed-deployment replay.

The tests exercise lexical and hashing-vector claim retrieval, real SQLite
FTS and hybrid passage indexes, both agent paths, single-item limits, explicit
intent overrides, missing source refs, and competing authoritative claims.
The existing project-memory retrieval evaluation also passes. Validation totals:
166 engine tests, 21 core intent tests, and 114 CLI/presentation tests passed
(301 total); Ruff and `git diff --check` passed. Independent review found no
remaining required changes.

Managed verification remains unavailable: `managed:pms` returns
`pot_not_found`, and the user confirmed PMS is absent in this environment.
The [managed check record](evaluations/definition-managed-check.json) captures
the JSON error and local revision/version information. The canonical feature
metadata, `DOCUMENTS` link, supporting chunk content, ingestion/linking history,
and deployed server revision still require inspection in the original
environment. No ingestion defect is attributed from this local evaluation.

### 7. Make feature-anchor behavior explicit (friction 8)

Confirmed: the reader treats anchors as the subject of `PROVIDES` or the object
of `IMPLEMENTED_IN`. A feature occupies the opposite endpoints. A repository
anchor returned results in the reproduction; the feature anchor returned none.

Next steps: support feature anchors by traversing the corresponding endpoints
and returning their implementation/documentation context, or reject unsupported
anchor types with clear repo/service guidance. Document the chosen contract.

Acceptance: a known feature anchor returns its context or a specific validation
error, rather than a misleading successful empty result. Repository and service
anchors retain their behavior.

Code: `potpie/context-engine/src/potpie_context_engine/application/readers/features.py`
and `potpie/context-core/src/potpie_context_core/graph_workbench_ontology.py`.

### 8. Finish selector and discovery ergonomics (frictions 3, 4)

Implementation status: explicit qualified pot failures now include confirmed
alternatives on other configured origins. `mutation-template` accepts an
optional `--pot` without resolving it or requiring a host connection.

Completed:

- An unreachable explicit `local:pms` remains an availability error (exit 2),
  while a reachable host without that pot returns `pot_not_found` (exit 1).
  Both can suggest `--pot managed:pms` after confirming a live matching pot.
  The command does not reroute or change the persisted active origin.
- Alternative lookup runs only after explicit targeting fails. Unconfigured,
  unreachable, missing, and archived alternatives produce no suggestion;
  diagnostic failures preserve the original error. Authentication refusals
  retain credential guidance (exit 4). Successful targeting does not probe
  other origins. Diagnostic enumeration uses the existing host RPC timeout.
- `graph mutation-template --kind feature --pot local:pms` works offline.
  The selector is ignored, not resolved or validated, and the output remains
  an unscoped schema skeleton. Command help and the main agent/Claude examples
  state this explicitly; the actual target is selected on `graph propose`.

Acceptance: origin errors distinguish unreachable from not-found, suggestions
name only confirmed live alternatives, and template discovery works without a
daemon or managed connection. Rerun the original commands through Windows
Copilot ACP after deploying the CLI changes.

Code: `potpie/cli/commands/_common.py`, `potpie/cli/commands/graph.py`, and
`potpie/cli/templates/`. Regressions cover human/JSON errors, origin isolation,
optional-lookup failures, and offline template generation.

Validation: 206 routing, ergonomics, error-contract, and agent-template tests
passed. Ruff lint, formatting, and `git diff --check` passed. Windows ACP
validation remains pending.

### Delivery and Verification

Implement encoding, commit recovery, filter validation, and threshold behavior
as separate focused changes. Investigate the managed source-reference failure
early; it blocks exact evidence inspection. Follow with retrieval/linking,
feature anchors, and selector ergonomics.

Review baseline: 16 relevant existing CLI/service tests passed. Additional
isolated probes reproduced the encoding/help failure, unsupported-filter success,
empty feature-anchor result, and ignored passage threshold. Source refs retained
their type through an RPC round trip. These checks do not replace Windows ACP
or managed-backend validation.

Before closing this report, record the shipped CLI, server, and extension build
revisions and rerun the original session commands through the Windows VS Code
Copilot ACP flow. Record the outcome against each acceptance criterion above.
