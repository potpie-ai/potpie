# Context Graph Docs

Last reviewed: 2026-05-29.

The Context Graph is Potpie's project-context layer for agents. Users and agents
talk to the `potpie` CLI. The same Pot Management, Graph Service, and Skill
Manager modules run inside either a local daemon or the managed API server. Graph
state stays local by default unless the user explicitly selects cloud graph sync
or a managed profile.

```mermaid
flowchart TB
  cg_cli["potpie CLI"]

  subgraph cg_local_profile["Local profile"]
    direction TB
    cg_local_daemon["local daemon"]
    cg_local_services["same service modules"]
    cg_local_store[("local stores")]
    cg_local_daemon --> cg_local_services --> cg_local_store
  end

  subgraph cg_managed_profile["Managed graph profile"]
    direction TB
    cg_managed_api["managed API server"]
    cg_managed_services["same service modules"]
    cg_hosted_store[("hosted stores")]
    cg_managed_api --> cg_managed_services --> cg_hosted_store
  end

  subgraph cg_event_ledger["Event Ledger"]
    direction TB
    cg_ledger["managed or self-hosted<br/>webhooks + event history + cursors"]
  end

  cg_cli --> cg_local_daemon
  cg_cli -. "cloud profile/sync" .-> cg_managed_api
  cg_cli -. "ledger config/pull" .-> cg_ledger
  cg_local_services -. "pull events" .-> cg_ledger
  cg_managed_services -. "consume events" .-> cg_ledger
```

## Start Here

| Doc | What it answers |
|---|---|
| [`vision.md`](./vision.md) | What are we building, and what are the product constraints? |
| [`architecture.md`](./architecture.md) | What are the pieces, runtime flows, agent contract, extension points, and implementation rules? |
| [`cli-flow.md`](./cli-flow.md) | What should the shared CLI journey and command contract look like across local and managed profiles? |
| [`observability.md`](./observability.md) | What should logs, traces, metrics, and readiness report? |
| [`bench-plan.md`](./bench-plan.md) | How do we validate graph quality across backends? |

## Target OSS Default

```bash
pip install potpie
potpie setup --repo . --agent claude --scan
potpie status
```

`potpie setup` installs/starts the daemon when needed, creates and uses a local
`default` pot on first run, registers the repo source, and can perform the first
scan. Users only pass `--pot <name>` when the first pot should have a different
name.

Cloud is opt-in and visibly scoped:

```bash
potpie cloud login
potpie cloud push
potpie cloud pull
potpie cloud status
```

Managed or self-hosted integration events are also opt-in:

```bash
potpie cloud login
potpie ledger use managed
potpie ledger pull --apply
```

This can feed a local graph from a managed Event Ledger without pushing graph
state to managed storage.

## Vocabulary

| Term | Meaning |
|---|---|
| **Pot** | Workspace/tenant boundary. Every query, source, record, claim, and graph operation is scoped to one pot. |
| **Daemon shell** | Local background process for lifecycle, auth, IPC, health, logs, and service hosting. It is not the business layer. |
| **Pot Management Service** | Control plane for pots, active pot, source registry, graph readiness, lifecycle, and export/import metadata. |
| **Graph Service** | Data plane for `resolve`, `search`, `record`, and `status`. Owns readers, ranking, record lowering, and envelopes. |
| **GraphBackend** | Swappable graph capability bundle: mutation, claim query, semantic search, inspection, analytics, snapshot. |
| **Skill Manager Service** | CLI-managed skill catalog and installation layer for agent harnesses. Skills teach agents how to use the CLI; they are not graph facts or new tools. |
| **Event Ledger** | Separate managed or self-hostable source-event service for webhooks, integration polling, event history, and cursors. Graphs pull/consume from it; it is not graph storage. |

The code currently lives under
[`app/src/context-engine/`](../../app/src/context-engine/). If docs conflict,
prefer this order: `vision.md`, `architecture.md`, then the operational docs.
