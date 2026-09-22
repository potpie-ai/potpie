<!-- potpie-start -->
Potpie is durable project memory for repository context, decisions, infrastructure,
bugs, documents, and preferences. Skip it for simple questions and trivial edits.

For non-trivial work, load the relevant `potpie-*` skill, then run one shared
discovery pass. Start with `potpie resolve "<task>" --pot <origin:name>`. For code
work, also run `potpie graph read --subgraph decisions --view
preferences_for_scope --repo current --limit 12 --pot <origin:name>`. Reuse the
selected pot and returned entity keys in follow-ups. Run `potpie status` only when
health needs checking.

Use `potpie-graph` for detailed read, write, identity, evidence, and reporting
guidance. Use the task-specific skills for debugging, architecture, timelines,
repository baselines, source ingestion, and document formats. Record only durable
learnings, and show the exact Potpie commands behind an answer.
<!-- potpie-end -->
