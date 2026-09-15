---
name: potpie-provenance
version: "1"
recommended: true
description: "Use when recording which chat prompt and spec produced generated code, linking file+line spans with IMPLEMENTS, or answering potpie why for a path."
---

# Potpie Provenance

Use this skill when you drafted a requirement, wrote or edited code from a
chat prompt, or need to answer which prompt/spec produced a file span.

Capture is **harness-led**. You decide the spec text and the `IMPLEMENTS`
link. Potpie stores hashed keys on the graph and full prompt/spec bodies in
SQLite. It does **not** scan the repo or auto-write lineage from a nudge.

`IMPLEMENTS` is CodeAsset → SpecRequirement. Do not confuse it with
`IMPLEMENTED_IN` (Feature → code).

## Record a spec (plan / Stop)

When you have a draft or final requirement:

```bash
potpie record --type spec_requirement --summary "<short spec>" \
  --scope repo:<owner/name>
```

Prefer the structured capture path so hashes stay aligned with the span index:

```bash
potpie lineage capture --spec "<requirement text>" --session "<session-id>" \
  --harness claude --path <file> --lines 12-40
```

Use `generation_link` when the code span already exists and you only need the
graph claim:

```bash
potpie record --type generation_link --summary "<path>:12-40"
```

Details the writer expects: `code_asset_key`, `prompt_key`, optional
`spec_key`. Do not paste full chat logs onto the graph.

## Query

```bash
potpie why path/to/file.py --lines 12-40
potpie graph read --subgraph provenance --view lineage --scope path:path/to/file.py
```

## Cursor / Codex

Those harnesses have no UserPromptSubmit / PostToolUse Write hooks. Record
manually at plan and Stop. Claude hooks store the latest prompt and link
Write/Edit spans fail-open; they never block an edit.
