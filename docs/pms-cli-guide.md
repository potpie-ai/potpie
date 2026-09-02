# Potpie CLI Guide for the PMS Pot

This guide uses the current managed Potpie project pot:

```text
Name: pms
pot_id: pot_d41f0f0451d74dc1
CLI scope: managed:pot_d41f0f0451d74dc1
```

All knowledge lookup examples below explicitly target this pot. Use the
`managed:` prefix when a command must not fall back to a local daemon.

```text
managed:pot_d41f0f0451d74dc1
```

## Current pot and explicit scope

`pot list` discovers available pots and `pot info` reports the active pot. The
`pot info` command has no `--pot` option because it describes the active pot.

Inspect or select the PMS pot and list its registered sources:

```powershell
potpie --json pot list
potpie pot use pot_d41f0f0451d74dc1
potpie --json pot info
potpie --json source list --pot managed:pot_d41f0f0451d74dc1
```

`pot use` is optional when the PMS pot is already active. For reproducible
commands, pass `--pot managed:pot_d41f0f0451d74dc1` explicitly to commands that
read project knowledge.

## Search and resolve project knowledge

Use `search` for a focused follow-up lookup:

```powershell
potpie search "PMS full form" --pot managed:pot_d41f0f0451d74dc1
potpie --json search "PMS-Service" --pot managed:pot_d41f0f0451d74dc1
```

Use `resolve` for a bounded answer with project context:

```powershell
potpie --json resolve "What does PMS stand for in this project?" --intent docs --mode deep --pot managed:pot_d41f0f0451d74dc1
```

The available intents include `onboarding`, `docs`, `operations`, and
`debugging`. Use `--json` when another tool needs structured output.

## Inspect the graph

Check graph availability and discover the applicable graph contract:

```powershell
potpie --json graph status --pot managed:pot_d41f0f0451d74dc1
potpie --json graph catalog --task "Understand the PMS service architecture" --pot managed:pot_d41f0f0451d74dc1
```

Describe a view before reading it:

```powershell
potpie --json graph describe infra_topology --view service_neighborhood --examples --pot managed:pot_d41f0f0451d74dc1
```

Read the selected view:

```powershell
potpie --json graph read --subgraph infra_topology --view service_neighborhood --scope service:pms-cas --limit 20 --pot managed:pot_d41f0f0451d74dc1
```

`status` checks graph readiness, `catalog` identifies supported operations and
views, `describe` explains a view and provides examples, and `read` returns
the project-memory results.

## Read document resources

List the sections and chunks in the PMS functional specification:

```powershell
potpie resource list --doc pms-functional-specification-va20a --pot managed:pot_d41f0f0451d74dc1
```

Read a chunk and its neighboring chunks:

```powershell
potpie resource get potpie://res/pms-functional-specification-va20a/references/0000 --with-neighbors --pot managed:pot_d41f0f0451d74dc1
```

`resource list` finds resource IDs. `resource get` reads the stored document
text without running another graph query.

## Troubleshooting scope

If the local Potpie daemon is unavailable, target the managed host explicitly:

```powershell
potpie --json search "PMS" --pot managed:pot_d41f0f0451d74dc1
```

The bare ID form, `--pot pot_d41f0f0451d74dc1`, can resolve to the local host;
the `managed:` prefix selects the managed host and avoids a local daemon
connection attempt.
