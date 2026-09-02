<!-- potpie-start -->
Potpie is durable project memory: repo/source mappings, decisions, infra,
changes, bugs, docs, and preferences for agents. Use it when it can materially
help with repo context, decisions, architecture, bugs, or durable history.
Do not run Potpie checks for simple Q&A or trivial edits. The health check is
one call, `potpie status`; stop only if the daemon is down or the pot is empty.
`potpie resolve "<task>"` is the first read, `potpie record` the first write;
record only durable learnings. When an answer used Potpie, show the `potpie`
commands behind it verbatim (empty reads in one line) and draw a mermaid diagram
only when the answer is a shape (`flowchart` for entities and edges, `timeline`
for ordered events), never for a single fact or a list.
<!-- potpie-end -->
