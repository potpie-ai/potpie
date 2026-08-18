<!-- potpie-start -->
Potpie is durable project memory: repo/source mappings, decisions, infra,
changes, bugs, docs, and preferences for agents. Use it when it can materially
help with repo context, prior decisions, architecture, bugs, or durable history.
Do not run Potpie checks for simple Q&A or trivial edits. When useful, check
mapping/graph health once per session (`potpie --json source list`,
`potpie --json graph status`; skip if unavailable). Record only durable learnings.
When an answer used Potpie, show the `potpie` commands behind it verbatim, and
draw a mermaid diagram when the answer is a shape rather than a fact
(`flowchart` for entities and their edges, `timeline` for an ordered run of
events) — never for a single fact or a plain list.
<!-- potpie-end -->
