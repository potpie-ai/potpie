# Optional: Hatchet as the context-graph queue

Potpie’s **default** context-graph job queue is **Celery** (`CONTEXT_GRAPH_JOB_QUEUE_BACKEND` unset or `celery`). The **Hatchet adapter** remains in the codebase for optional use.

## When to use Hatchet

Set **`CONTEXT_GRAPH_JOB_QUEUE_BACKEND=hatchet`**, install **`hatchet-sdk`**, set **`HATCHET_CLIENT_*`** (see `hatchet_sdk.config.ClientConfig`), and run a **Hatchet worker**:

```bash
python -m app.modules.context_graph.hatchet_worker
```

Producers use **`HatchetContextGraphJobQueue`** (`event.push`) via `bootstrap.queue_factory.get_context_graph_job_queue()`.

## Self-hosting Hatchet

Potpie does **not** ship Hatchet in root **`compose.yaml`**. Use upstream **[Hatchet Lite](https://docs.hatchet.run/self-hosting/hatchet-lite)** — copy their Postgres + `hatchet-lite` compose snippet into a separate file or merge into your own stack — then mint a JWT (e.g. `hatchet-admin token create` per [self-hosting docs](https://docs.hatchet.run/self-hosting)) and set **`HATCHET_CLIENT_TOKEN`**.

### Why a JWT?

Hatchet is a control plane (HTTP + gRPC), not a raw broker like Redis. The Python SDK authenticates with a JWT whether you use Hatchet Cloud or self-hosted Lite. The token is issued by **your** Hatchet instance (not a requirement to use Hatchet Cloud).

## References

- [Hatchet v1](https://docs.hatchet.run/v1), [Hatchet Lite](https://docs.hatchet.run/self-hosting/hatchet-lite), [Self-hosting](https://docs.hatchet.run/self-hosting)
