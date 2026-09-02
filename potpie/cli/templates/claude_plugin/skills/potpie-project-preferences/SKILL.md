---
name: potpie-project-preferences
version: "2"
description: "Use before writing, modifying, reviewing, refactoring, or testing code so repo/project preferences surface: error handling, file structure, frameworks, logging, dependency choices, testing, security, API style, and naming. Also use after code work when a reusable project preference should be recorded."
---

# Potpie Project Preferences

Use this skill before implementation or review so local conventions shape the
work instead of being rediscovered from code.

## Fast Path

Two reads, in this order. Every read header names the pot; once you know it,
pass `--pot local:<name>` on later calls.

1. One bounded context call — decisions, features, infra and prior bugs as
   `subject PREDICATE object` triples:

```bash
potpie resolve "<the task in the user's words>"
```

2. Preferences by scope, with **no `--query`**. Preferences are constraints,
   not search hits: this view applies an absolute 0.7 similarity floor to a
   query, which a task sentence rarely clears, and `resolve` applies the same
   floor to the task text, so it lists a preference only on a near-verbatim
   match. This read is not optional.

```bash
potpie graph read --subgraph decisions --view preferences_for_scope --repo current --limit 12
potpie graph read --subgraph decisions --view preferences_for_scope --scope service:<service>,path:<path-or-dir> --limit 12
```

`--repo current` takes the repo key from the working tree; spelled by hand it
is `repo:<host>/<org>/<name>` (`repo:github.com/acme-corp/acme-shop`). A scope
hides only preferences bound to a *different* value of the same dimension: a
service key hides other services' rules, a repo key hides nothing bound by
service, and `items=0` means nothing is bound to that value. Each row comes
with its anchor entity, so `items=2` is one preference. Add
`--query "<terms>" --query-threshold 0.4` only when the scope returns more
than a dozen rows. Search only for a key no read has shown:
`potpie graph search-entities "<repo service package>" --limit 10`. Do not
pre-read `pot info`, `source list`, or `graph status`.

## Apply Results

Treat returned preferences as implementation constraints. Prefer active,
higher-confidence, closer-scope preferences over broad ones: file, directory,
service, repo, global. If two preferences conflict, verify source refs or ask the
user before choosing.

Do not quote Potpie context back unless it matters. Use it to write better code.

## Report Back

Still show the `graph read` you ran, with its `--scope` or `--repo`. That is
not the same as quoting the context back: the command is how the reader checks
that you looked at the right scope, while the preference text belongs in the
code you wrote, not in the reply. Say in one line when a read came back empty —
code written against no preferences is a different claim from code written
against none that applied.

Preferences are a list, not a shape. Report them as prose or a short table and
do not draw a mermaid diagram; boxes around a scope hierarchy are slower to read
than the sentence "file beats directory beats service beats repo".

## Record A Preference

Record only reusable, explicit preferences that are likely to matter again. Do
not turn one-off implementation choices into project policy.

One preference is one call — no JSON file, no plan:

```bash
potpie record --type preference --summary "<one-line prescription>" --detail policy_kind=<error_handling|logging|testing|library_choice|file_structure> --detail prescription="<specific guidance an agent should follow>" --scope service:<service>
```

Optional details: `strength=<hard|strong|soft>`, `audience=<team|service|project|global>`.
`--scope` takes an existing key; an unknown one mints a stub, so reuse a key a
read returned. The reply is a `record_id` and the mutation count; the entity
key is minted from the whole `prescription`, so keep it short and lead with
the distinctive words.

The plan flow is for a batch — several preferences, or one tied to a
`CodeAsset` with evidence:

```bash
potpie graph mutation-template --kind preference-policy
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
```

The template carries the keys, predicates and required properties; `propose`
validates the rest. Omit `graph_contract_version` from the payload.

A good preference write includes the policy kind, prescription, strength,
audience, scope, truth class, evidence or source refs when available, and a
retrieval-grade description with the terms future agents would search.

Preference capture is harness-led: read the source yourself, then write semantic
facts. Do not use scanner-driven graph updates or infer policy from code shape
alone.
