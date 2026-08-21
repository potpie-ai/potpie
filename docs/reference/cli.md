---
title: CLI reference
description: Main Potpie CLI commands and examples.
---

## Basic CLI user checklist

The main CLI commands are:

| Command | Purpose |
| --- | --- |
| `potpie setup` | Run first-time local setup for config, daemon, default pot, and agent skills. |
| `potpie login` | Sign in to Potpie for account-backed and managed features. |
| `potpie github login` / `potpie linear login` | Connect source integrations you want agents to use. |
| `potpie status` | Show context readiness for the active pot, including daemon, graph, and skill checks. |
| `potpie auth status` | Show configured integration auth status. |
| `potpie auth status --verify` | Verify integration credentials with lightweight API checks. |
| `potpie doctor` | Run local diagnostics for daemon, backend capabilities, and skill drift. |
| `potpie source add repo .` | Register the current repo as a source for the resolved pot. |
| `potpie pot list` / `potpie pot use {id-or-name}` | List pots and choose the active workspace. |
| `potpie resolve "{task}"` | Pull the context an agent should read before doing a task. |
| `potpie search "{query}"` | Look up a specific file, workflow, bug, decision, or convention. |
| `potpie record --type {type} --summary "{summary}"` | Write a durable project learning. |
| `potpie graph ...` | Use lower-level graph reads, quality checks, proposals, and commits. |
| `potpie ui` | Open the local graph explorer served by the daemon. |
| `potpie skills install --agent {agent}` | Install or refresh Potpie guidance for an agent harness. |

Examples:

```bash
potpie setup --repo . --agent claude
potpie github login
potpie auth status
potpie status
potpie source add repo .
potpie resolve "what should I know before working in this repository?"
potpie search "authentication flow"
potpie record --type decision --summary "Prefer the Potpie CLI for graph work"
```

You can find an exhaustive list with more examples in our [docs](https://docs.potpie.ai).
