# Queued Task Flow

This maps the request-to-queue-to-worker path for queued runs in this repo. It is
intended for debugging runs that stay `queued`, never appear in the stream, or
are accepted by the API but never show worker progress.

## Main Debugging Target

For conversation runs, the critical path is:

```text
POST /conversations/{conversation_id}/message/?stream=true
  -> start_celery_task_and_stream(...)
  -> Redis task metadata + queued stream event
  -> execute_agent_background.delay(...)
  -> Celery queue: ${CELERY_QUEUE_NAME}_agent_tasks
  -> worker subscribed with -Q ${CELERY_QUEUE_NAME}_agent_tasks
  -> execute_agent_background(...)
  -> Redis task status running + chat stream chunks/end
```

If a run remains `queued`, isolate which handoff above did not happen.

## Broker and Queue Routing

Celery uses Redis for both broker and result backend. The Redis URL is built from
`REDISHOST`, `REDISPORT`, `REDISUSER`, and `REDISPASSWORD` in
`app/celery/celery_app.py`.

The queue prefix comes from `CELERY_QUEUE_NAME`, defaulting to `staging`.

Confirmed task routes:

| Task | Queue |
| --- | --- |
| `app.celery.tasks.parsing_tasks.process_parsing` | `${CELERY_QUEUE_NAME}_process_repository` |
| `app.celery.tasks.parsing_tasks.process_colgrep_index` | `${COLGREP_INDEX_QUEUE_NAME:-${CELERY_QUEUE_NAME}_colgrep_index}` |
| `app.celery.tasks.agent_tasks.execute_agent_background` | `${CELERY_QUEUE_NAME}_agent_tasks` |
| `app.celery.tasks.agent_tasks.execute_regenerate_background` | `${CELERY_QUEUE_NAME}_agent_tasks` |
| `app.modules.event_bus.tasks.event_tasks.process_webhook_event` | `external-event` |
| `app.modules.event_bus.tasks.event_tasks.process_custom_event` | `external-event` |

Notes:

- The event task decorators name `external-event-webhook` and
  `external-event-custom`, but `app/celery/celery_app.py` routes both task names
  to `external-event`, and `CeleryEventBus.send_task(...)` also explicitly
  publishes to `external-event`.
- `COLGREP_INDEX_QUEUE_NAME` overrides only the ColGREP index queue.

## Agent Run Flow

1. `POST /conversations/{conversation_id}/message/` in `app/api/router.py`
   validates the message, checks usage, and normalizes the run id.
2. Without a replay `cursor`, the API reserves a unique run id with
   `async_ensure_unique_run_id(...)`.
3. For `stream=true`, the API calls `start_celery_task_and_stream(...)`.
   For `stream=false`, it calls `start_celery_task_and_wait(...)`. Both use the
   same queue handoff.
4. The routing helper sets `task:status:{conversation_id}:{run_id}` to
   `queued`.
5. The helper publishes a `queued` event to
   `chat:stream:{conversation_id}:{run_id}`.
6. The helper enqueues `execute_agent_background.delay(...)`.
7. Celery routes `execute_agent_background` to
   `${CELERY_QUEUE_NAME}_agent_tasks`.
8. The helper stores the returned Celery id in
   `task:id:{conversation_id}:{run_id}`.
9. The API waits up to 30 seconds for `task:status` to become one of `queued`,
   `running`, `completed`, or `error`, then returns/collects the Redis stream.
10. A worker subscribed to `${CELERY_QUEUE_NAME}_agent_tasks` picks up the task.
11. `execute_agent_background` logs `Starting background agent execution` and
    sets `task:status:{conversation_id}:{run_id}` to `running`.
12. The worker publishes lifecycle, chunk, and terminal `end` events back to
    `chat:stream:{conversation_id}:{run_id}`.

Confirmed handoff points:

- API -> Redis status: `task:status:{conversation_id}:{run_id}=queued`
- API -> Redis stream: `chat:stream:{conversation_id}:{run_id}` receives
  `type=queued` with `task_id`, `queue_name`, `queue_depth_at_enqueue`, and
  `enqueued_at`
- API -> Celery broker: `execute_agent_background.delay(...)`
- API -> Redis task id: `task:id:{conversation_id}:{run_id}=<celery task id>`
- API -> Redis enqueue metadata:
  `task:enqueue:{conversation_id}:{run_id}` stores the task id, queue name,
  enqueue timestamp, and queue depth at enqueue
- Celery broker -> worker: worker receives task from
  `${CELERY_QUEUE_NAME}_agent_tasks`
- Worker -> Redis status: `task:status:{conversation_id}:{run_id}=running`
- Worker -> Redis stream: `type=running` includes `queue_depth_at_enqueue`,
  `queue_depth_at_pickup`, and `pickup_latency_ms`; chunks and terminal `end`
  follow

Important files:

- `app/api/router.py`
- `app/modules/conversations/utils/conversation_routing.py`
- `app/celery/tasks/agent_tasks.py`
- `app/modules/conversations/utils/redis_streaming.py`
- `app/celery/celery_app.py`

## Regenerate Flow

Regeneration follows the same Redis status/stream handoff as a normal agent run,
but the API enqueues `execute_regenerate_background.delay(...)`.

It routes to the same queue:

```text
${CELERY_QUEUE_NAME}_agent_tasks
```

One behavioral difference: the regenerate API waits for `task:status` to become
`running`, not merely `queued`, before considering the start check successful.

Important files:

- `app/modules/conversations/conversations_router.py`
- `app/celery/tasks/agent_tasks.py`

## Repository Parsing Flow

1. `POST /parse` calls `ParsingController.parse_directory(...)`.
2. The parsing controller registers or updates a project and sets it to
   `submitted`.
3. The controller enqueues `process_parsing.delay(...)`.
4. Celery routes `process_parsing` to
   `${CELERY_QUEUE_NAME}_process_repository`.
5. A worker subscribed to that queue picks up the task.
6. `process_parsing` logs `Task received: Starting parsing process`, creates
   `ParsingService`, and runs `parse_directory(...)`.
7. During parsing, ColGREP indexing may be scheduled with
   `process_colgrep_index.delay(...)`.
8. Celery routes `process_colgrep_index` to
   `${COLGREP_INDEX_QUEUE_NAME:-${CELERY_QUEUE_NAME}_colgrep_index}`.
9. If ColGREP enqueue fails, parsing falls back to a local background thread for
   index build.

Important files:

- `app/api/router.py`
- `app/modules/parsing/graph_construction/parsing_controller.py`
- `app/modules/parsing/graph_construction/parsing_service.py`
- `app/celery/tasks/parsing_tasks.py`
- `app/celery/celery_app.py`

## Event Bus Flow

Webhook/custom events are not conversation runs, but they share the same Celery
broker. They are useful when checking whether all workers or only agent workers
are stuck.

1. `CeleryEventBus.publish_webhook_event(...)` or
   `publish_custom_event(...)` builds the event payload.
2. It calls `celery_app.send_task(...)` with `queue="external-event"`.
3. Celery routes `process_webhook_event` and `process_custom_event` to
   `external-event`.
4. A worker started by `scripts/start_event_worker.sh` consumes
   `external-event`.

Important files:

- `app/modules/event_bus/celery_bus.py`
- `app/modules/event_bus/tasks/event_tasks.py`
- `scripts/start_event_worker.sh`
- `app/celery/celery_app.py`

## Worker Pickup

Stage and production Celery worker configs subscribe to both main queues:

```text
${CELERY_QUEUE_NAME}_process_repository,${CELERY_QUEUE_NAME}_agent_tasks
```

They also start a separate ColGREP worker for:

```text
${COLGREP_INDEX_QUEUE_NAME:-${CELERY_QUEUE_NAME}_colgrep_index}
```

Local `scripts/start.sh` follows the same split.

One mismatch to watch: the root `supervisord.conf` starts a worker only for
`${CELERY_QUEUE_NAME}_process_repository`. If that config is used by itself,
agent tasks can be enqueued but never picked up because no worker is subscribed
to `${CELERY_QUEUE_NAME}_agent_tasks`.

Also note `app/celery/worker.py`: without `-Q`, Celery consumes the default
`celery` queue, not the routed Potpie queues. Start workers with explicit queue
subscriptions.

## Likely Stuck Points

- Redis connection or env mismatch: API and worker must resolve the same Redis
  host, port, credentials, and database.
- Queue prefix mismatch: API and worker must use the same `CELERY_QUEUE_NAME`.
  A task enqueued to `staging_agent_tasks` will not be consumed by a worker
  listening on another prefix.
- Worker queue subscription mismatch: a worker started without
  `${CELERY_QUEUE_NAME}_agent_tasks` will not pick up agent or regenerate runs.
- Worker started without `-Q`: the worker may listen only on the default
  `celery` queue and never receive routed Potpie tasks.
- Root `supervisord.conf` only listens to the parsing queue, so it is not enough
  for conversation agent runs.
- Missing ColGREP worker: parsing can complete while ColGREP index builds remain
  queued if no worker listens on the ColGREP queue.
- Task accepted but status stays `queued`: the API successfully enqueued and
  stored the Celery id, but no worker has run far enough to set status to
  `running`.
- `task:id` missing after the API returned queued: the `.delay(...)` call may
  have failed, or Redis failed before the returned Celery id was stored.
- Stream has queued event but no chunks/end: worker pickup, worker crash, or
  worker Redis publish path is the likely boundary.
- High `queue_depth_at_enqueue` plus increasing pickup latency means backlog.
- Low/zero queue depth at enqueue but no `running` event usually points to queue
  routing or worker subscription mismatch.
- `running` event exists with high `pickup_latency_ms` and low
  `queue_depth_at_pickup` means the task was delayed before pickup but the
  backlog had drained by the time this worker started it.
- Task status key expiration: `task:status:*` and `task:id:*` expire after 10
  minutes, so older stuck runs may lose the Redis breadcrumbs used for debugging
  or cancellation.
- Worker crash or hard kill: Celery uses late ack and a visibility timeout tied
  to `CELERY_TASK_TIME_LIMIT`, but `task_reject_on_worker_lost=False` means a
  lost worker may not requeue indefinitely.
- Parsing-specific recovery exists only for projects stuck in `submitted`; the
  parsing status endpoint can re-submit after
  `PARSING_STUCK_THRESHOLD_MINUTES`, defaulting to 10 minutes.

## Fast Checks

Use these checks when a queued run stops progressing:

1. Confirm the run's Redis metadata:
   `task:status:{conversation_id}:{run_id}` and
   `task:id:{conversation_id}:{run_id}`.
2. Confirm enqueue diagnostics:
   `task:enqueue:{conversation_id}:{run_id}`.
3. Confirm the stream exists and has more than a queued event:
   `chat:stream:{conversation_id}:{run_id}`.
4. Check `queue_depth_at_enqueue`, `queue_depth_at_pickup`, and
   `pickup_latency_ms` on the queued/running stream events.
5. Confirm the expected queue from the task name in `app/celery/celery_app.py`.
6. Confirm a worker is subscribed to the expected queue with the same
   `CELERY_QUEUE_NAME`.
7. Check worker logs for `Task received`, `Starting background agent execution`,
   or `Starting background regenerate execution` using the Celery task id or
   `conversation_id:run_id`.
8. If Celery shows the task active but Redis status is still `queued`, inspect
   worker logs before `redis_manager.set_task_status(..., "running")`.
9. If Redis status is `running` but no stream chunks arrive, inspect agent/tool
   execution and Redis publish failures in the worker logs.

## Definition-of-Done Checklist

- [x] **Request-to-queue-to-worker path documented**: agent, regenerate,
  parsing, ColGREP, and event-bus paths are mapped above.
- [x] **Queue names and handoff points confirmed**: Celery routes, Redis status
  keys, stream keys, task-id keys, and worker `-Q` subscriptions are listed.
- [x] **Queue depth and pickup timing visible**: queued/running stream events,
  enqueue metadata, and worker logs include queue depth and pickup latency.
- [x] **Likely stuck points identified**: queue/env mismatches, missing worker
  subscriptions, Redis metadata gaps, status transitions, TTL expiry, and worker
  crash behavior are called out above.
