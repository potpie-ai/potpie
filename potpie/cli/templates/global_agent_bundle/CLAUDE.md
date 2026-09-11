<!-- potpie-start -->
Potpie is durable project memory: repo/source mappings, decisions, infra,
changes, bugs, docs, and preferences for agents. Use it when it can materially
help with repo context, decisions, architecture, bugs, or durable history.
Do not run Potpie checks for simple Q&A or trivial edits. Share one discovery
pass across relevant skills: run `potpie resolve "<task>"`, scope-only
`potpie graph read --subgraph decisions --view preferences_for_scope --repo current`
for code work (no `--query`), and untyped `potpie graph search-entities "<name>"`
for named entities with unknown keys concurrently.
Skip inapplicable branches and reuse current reads or hook context for the same
task, pot, and scope. Run `potpie status` alongside reads if health needs checking.
Use a known explicit pot selector; resolve ambiguous routing first and check pot
IDs before combining results. Follow returned keys with needed neighborhoods or
views and chunk IDs with batched `resource get`; parallelize those follow-ups
once inputs are known. Stop when evidence and applicable constraints are covered.
Before ingestion, inspect
`potpie graph catalog --profile full` and follow `potpie-graph` ontology
selection: classify facts, decisions, preferences, and events before choosing
a writer. Use `record` for supported structured learnings and semantic plans
for topology/features even for one fact. Record only durable learnings. When an answer used Potpie, show the `potpie`
commands behind it verbatim (empty reads in one line) and draw a mermaid diagram
only when the answer is a shape (`flowchart` for entities and edges, `timeline`
for ordered events), never for a single fact or a list.
<!-- potpie-end -->
