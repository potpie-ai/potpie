---
name: potpie-debug-memory
description: "Use while debugging or troubleshooting failures, flaky tests, incidents, production alerts, CI failures, local dev setup issues, repeated bugs, prior fixes, failed attempts, and verification history."
---

# Potpie Debug Memory

Use this skill before digging into a failure so prior symptoms, known fixes, and
failed attempts can guide the investigation.

## Fast Path

Search by symptom first, not just component name. Include exact error text,
commands, failing tests, environment, service, dependency, and synonyms.

```bash
potpie graph read \
  --subgraph debugging \
  --view prior_occurrences \
  --query "<expanded symptom query>" \
  --scope service:<service-name> \
  --limit 12
```

If no service is known, omit `--scope`. If the failure smells like a regression,
correlate with the timeline:

```bash
potpie graph read \
  --subgraph recent_changes \
  --view timeline \
  --format table \
  --time-window 7d \
  --query "<symptom feature dependency>" \
  --limit 20
```

If dependencies, adapters, environments, or deploys matter, read infra too:

```bash
potpie graph read \
  --subgraph infra_topology \
  --view service_neighborhood \
  --scope service:<service-name> \
  --depth 2 \
  --direction both
```

## Apply Results

Treat prior fixes as leads. Check whether the same symptom, environment,
version, dependency, data shape, command, or test path matches this incident.
Failed prior attempts are useful because they prevent repeated work.

## Record Debug Memory

Record after the investigation when the learning is reusable: bug pattern, fix,
verification, failed attempt, incident summary, runbook note, or setup gotcha.

Use the workbench write flow:

```bash
potpie --json graph catalog --task "record bug fix"
potpie graph search-entities "<service or symptom>" --limit 10
potpie --json graph describe debugging --view prior_occurrences --examples
potpie --json graph propose --file mutation.json
potpie --json graph commit <plan_id> --verify
potpie --json graph history --plan <plan_id>
```

Good debug memory includes the distinctive error text, repro signal, root cause
or uncertainty, fix steps, verification status, scope, truth class, evidence, and
a retrieval-grade description with symptom synonyms.

If the source may matter but the canonical update is uncertain, use
`potpie --json graph inbox add` instead of committing a weak fact.

Debug memory is harness-led: investigate and verify before writing. Do not use
scanner-driven graph updates or record a bug/fix from filenames or logs alone.
