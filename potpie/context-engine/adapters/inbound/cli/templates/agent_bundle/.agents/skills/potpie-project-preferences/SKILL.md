---
name: potpie-project-preferences
description: "Use before writing, modifying, reviewing, refactoring, or testing code so repo/project preferences surface: error handling, file structure, frameworks, logging, dependency choices, testing, security, API style, and naming. Also use after code work when a reusable project preference should be recorded."
---

# Potpie Project Preferences

Use this skill before implementation or review so local conventions shape the
work instead of being rediscovered from code.

## Fast Path

1. Identify the narrowest scope you know: repo, path, service, package, or file.
2. Expand the user's task into preference search terms: error handling, retries,
   validation, logging, observability, framework, folder layout, tests,
   dependency choice, security, API shape, naming.
3. Read preferences with the graph workbench:

```bash
potpie graph read \
  --subgraph decisions \
  --view preferences_for_scope \
  --scope repo:<owner-repo>,path:<path-or-dir> \
  --query "<expanded preference query>" \
  --limit 12
```

If the scope is unclear, first use `potpie --json pot info` and
`potpie --json source list`, or search entities:

```bash
potpie graph search-entities "<repo service package>" --type Service --limit 10
```

## Apply Results

Treat returned preferences as implementation constraints. Prefer active,
higher-confidence, closer-scope preferences over broad ones: file, directory,
service, repo, global. If two preferences conflict, verify source refs or ask the
user before choosing.

Do not quote Potpie context back unless it matters. Use it to write better code.

## Record A Preference

Record only reusable, explicit preferences that are likely to matter again. Do
not turn one-off implementation choices into project policy.

Use the workbench write flow:

```bash
potpie --json graph catalog --task "record coding preference"
potpie graph search-entities "<scope>" --type Service --limit 10
potpie --json graph describe decisions --view preferences_for_scope --examples
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

A good preference write includes the policy kind, prescription, strength,
audience, scope, truth class, evidence or source refs when available, and a
retrieval-grade description with the terms future agents would search.

Preference capture is harness-led: read the source yourself, then write semantic
facts. Do not use scanner-driven graph updates or infer policy from code shape
alone.

## MCP Fallback

Use this only when the `potpie` CLI is unavailable:

```json
{"intent":"feature","include":["coding_preferences","decisions","docs"],"mode":"fast","source_policy":"references_only"}
```
