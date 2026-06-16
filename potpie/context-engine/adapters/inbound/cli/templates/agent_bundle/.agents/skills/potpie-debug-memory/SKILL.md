---
name: "potpie-debug-memory"
version: "4"
recommended: true
description: "Use while debugging or troubleshooting so prior bugs, failed attempts, fixes, verifications, incidents, and dev setup gotchas surface before investigation."
---

# Potpie Debug Memory

Use this skill when investigating failures, flaky tests, incidents, production
alerts, local dev setup problems, CI failures, or recurring support issues.

Debug memory is harness-led: you decide what was learned, create a validated
plan with `graph propose`, and commit that plan with `graph commit`. Potpie
validates and stores. Nothing scans logs or the working tree into the graph
deterministically.

## Read First

Search by symptom, not just by component name:

```bash
potpie --json graph read \
  --subgraph debugging \
  --view prior_occurrences \
  --query "refund timeout deadlock concurrent settle retry flaky test" \
  --scope service:<service-name> \
  --limit 12
```

Then correlate recent changes and topology:

```bash
potpie --json graph read --subgraph recent_changes --view timeline --time-window 7d --limit 20
potpie --json graph read --subgraph recent_changes --view timeline --scope service:<service-name> --time-window 7d --limit 20
potpie --json graph read --subgraph infra_topology --view service_neighborhood --scope service:<service-name> --depth 2 --direction both
```

MCP compatibility fallback when shell access is unavailable:

```json
{"intent":"debugging","include":["prior_bugs","infra_topology","timeline"],"mode":"fast","source_policy":"references_only"}
```

## Use The Result

Treat prior fixes as leads. Check whether the same symptom, environment, version,
dependency, data shape, or test path matches this incident. Failed prior attempts
are as valuable as successful fixes because they prevent repeat work.

## Record Debug Memory

Record after the investigation when the learning is reusable:

- `bug_pattern`: symptom, scope, reproduction, and aliases.
- `fix`: exact steps that resolved it, root cause, verification status.
- `verification`: confirmation that a fix worked or did not work.
- `investigation` / `incident_summary`: useful narrative when no structured fix
  exists yet.
- `runbook_note` / `workflow`: repeatable troubleshooting steps.

Write reusable debug memory through the V2 plan workflow:

```bash
potpie --json graph search-entities "<service or symptom>" --type Service --limit 10
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id>
potpie --json graph history --plan <plan_id>
```

If only MCP is configured, `context_record` can capture compact debug records:

```json
{
  "record_type": "bug_pattern",
  "summary": "Payments settlement deadlocks when refund and settle run concurrently.",
  "details": {
    "kind": "runtime",
    "symptom_signature": "refund race timeout, payment deadlock on concurrent settle, postgres lock timeout",
    "scope_kind": "service",
    "reproduction_steps": ["Run refund and settle for the same order concurrently under load"]
  },
  "source_refs": ["github:issue:881"]
}
```

```json
{
  "record_type": "fix",
  "summary": "Order payment locks by account id before settle/refund writes.",
  "details": {
    "symptom_signature": "refund race timeout, payment deadlock on concurrent settle, postgres lock timeout",
    "fix_steps": ["Acquire account lock before order lock", "Keep retry budget below request timeout", "Verify with concurrent refund load test"],
    "root_cause": "Inconsistent lock acquisition order between refund and settle paths",
    "verification_status": "verified",
    "scope_kind": "service"
  },
  "source_refs": ["github:pr:912", "ci:run:4471"]
}
```

Write retrieval-grade summaries: include the error text, symptom synonyms,
environment, failing command/test, service, and the fix terms a future agent would
search for. Pick the truth class honestly (`agent_claim` for your own inference,
`source_observation` for what a source showed) and keep evidence refs
(`github:pr:…`, `ci:run:…`) on every fix and verification.
