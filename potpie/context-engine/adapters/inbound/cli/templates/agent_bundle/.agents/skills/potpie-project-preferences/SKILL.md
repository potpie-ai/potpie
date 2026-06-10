---
name: "potpie-project-preferences"
version: "2"
recommended: true
description: "Use before writing or reviewing code so project preferences surface: error handling, file structure, libraries, frameworks, logging, testing, security, and coding guidelines. Also use after work when a reusable preference should be recorded."
---

# Potpie Project Preferences

Use this skill when the task asks the agent to write, modify, refactor, review,
or test code. Project preferences should surface before implementation so the
agent follows local conventions instead of rediscovering them.

Preference capture is harness-led: you decide a preference exists and write it
through `context_record` / `graph mutate`; Potpie validates and stores. No
scanner derives preferences from the working tree.

## Read First

Prefer the graph CLI when shell is available:

```bash
potpie --json graph read \
  --view preferences.active_preferences \
  --scope repo:<owner-repo>,path:<path-or-dir> \
  --query "error handling logging retries framework testing file structure" \
  --limit 12
```

MCP equivalent:

```json
{"intent":"feature","include":["coding_preferences","decisions","docs"],"mode":"fast","source_policy":"references_only"}
```

Expand the query with the task's likely policy vocabulary: exception handling,
retry/backoff, validation, logging, observability, framework, folder layout,
test style, dependency choice, security, API shape, and naming.

## Apply The Result

Use returned preferences as implementation constraints, not as suggestions to
quote. Prefer active, higher-confidence, closer-scope preferences over generic
ones: file > directory > service > repo > global. If preferences conflict, verify
against source refs or ask the user before choosing.

## Record A New Preference

Record only preferences that are reusable, explicit, and likely to matter again.
Do not turn one-off implementation choices into policy.

Use `context_record` for simple captures:

```json
{
  "record_type": "preference",
  "summary": "External HTTP clients in payments should use bounded retry with structured timeout logging.",
  "details": {
    "policy_kind": "error_handling",
    "prescription": "Wrap payments outbound HTTP calls in the repo-standard retry helper with bounded attempts, timeout logging, and no broad exception swallowing.",
    "code_scope": {"repo": "acme/payments", "service": "payments-api", "language": "python"},
    "strength": "strong",
    "audience": "service",
    "justification_ref": "github:pr:412"
  },
  "source_refs": ["github:pr:412"]
}
```

Use `graph mutate` when linking the preference to a resolved scope yourself.
Resolve the target first:

```bash
potpie --json graph search-entities "payments api" --type Service --limit 10
```

Then write a retrieval-grade preference claim with `POLICY_APPLIES_TO`, truth
`preference`, and evidence when the preference came from a user, PR, ADR, doc, or
review comment.

Every durable write carries: a compact `summary` (display/browse), a
retrieval-grade `description` written for search (the situations, synonyms,
and scope a future agent would type), an honest truth class, and evidence
refs for anything stated by a source.
