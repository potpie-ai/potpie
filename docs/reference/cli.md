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
| `potpie resolve "{task}"` | Get the context an agent should read before starting a task. |
| `potpie search "{query}"` | Look up a specific file, workflow, bug, decision, or convention. |
| `potpie record --type {type} --summary "{summary}"` | Save a project decision, fix, or preference. |
| `potpie graph ...` | Use lower-level graph reads, quality checks, proposals, and commits. |
| `potpie ui` | Open the graph explorer served by the local daemon (`--pot managed:{id}` for a managed host). |
| `potpie host set {url} --token -` | Point the CLI at a managed context-graph host; the API key is read from stdin. |
| `potpie host use local` / `potpie host use managed` | Choose which host commands talk to by default. |
| `potpie host list` / `potpie host clear` | Show the configured hosts, or forget the managed one. |
| `potpie --host managed {command}` | Run one command against a host without changing the active one. |
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

### Managed hosts

A managed host serves the same commands from a hosted graph. Pot refs can name
their host (`managed:{id}`, `local:{id}`); a bare ref is looked up on the active
host only.

```bash
printf '%s\n' "$KEY" | potpie host set https://your-host.example --token -
potpie host use managed
potpie pot list --managed
potpie pot use managed:{id} --also-default-for-current-repo
potpie resolve "what changed in the payment flow?"
potpie --host local status
```

`potpie pot grant`, `revoke` and `grants` share a managed pot with other
people; the local host is single-user. Setup, skills, the daemon and
integration logins always stay on your machine.

See the [docs](https://docs.potpie.ai) for the full command reference and more examples.
