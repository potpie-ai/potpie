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
| High | `--source-ref` CLI/backend type mismatch |
| Medium | Broken top-level help |
| Medium | Misleading managed-versus-local pot error |
| Medium | Generic acronym lookup ranking |
| Medium | Unsupported filters accepted by generic `graph read` |
| Low | Inconsistent `--pot` support on schema-only commands |
| Low | Feature anchor behavior is narrower than its name suggests |

