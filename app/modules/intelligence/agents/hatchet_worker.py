"""Potpie Hatchet worker for agent runs (optional durable backend).

Run (after `make hatchet-up` writes .env.hatchet.local):

    make hatchet-worker
    # or: uv run python -m app.modules.intelligence.agents.hatchet_worker

Executes the same agent turn as the Celery backend via the shared ``run_agent_turn``,
publishing to the same Redis stream (``RedisStreamSink``) so the API's existing
streaming + cursor replay and the cooperative Redis cancellation work unchanged.
"""

from __future__ import annotations

from datetime import timedelta

from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.intelligence.agents.runtime.agent_execution_service import (
    run_agent_turn,
)
from app.modules.intelligence.agents.runtime.hatchet_backend import (
    AGENT_RUN_OPERATION_MESSAGE,
    AGENT_RUN_OPERATION_REGENERATE,
    EVENT_AGENT_RUN,
    AgentRunInput,
    prepare_hatchet_client_env,
)
from app.modules.intelligence.agents.runtime.redis_sink import RedisStreamSink
from app.modules.intelligence.provider.openrouter_usage_context import (
    get_and_clear_usages,
    init_usage_context,
)
from app.modules.intelligence.tracing.logfire_tracer import logfire_trace_metadata
from app.modules.utils.logger import log_context, setup_logger

logger = setup_logger(__name__)


def _resolve_user_email(db, user_id: str) -> str:
    """Best-effort sync email lookup (empty string fallback)."""
    from app.modules.users.user_service import UserService

    try:
        user = UserService(db).get_user_by_uid(user_id)
        return (getattr(user, "email", None) or "") if user else ""
    except Exception:
        logger.warning("Could not resolve email for user_id=%s", user_id)
        return ""


def _resolve_project_id(db, conversation_id: str):
    """Best-effort sync project_id lookup so logfire spans carry it (None fallback)."""
    from app.modules.conversations.conversation.conversation_model import Conversation

    try:
        conv = (
            db.query(Conversation).filter(Conversation.id == conversation_id).first()
        )
        if conv and conv.project_ids:
            return conv.project_ids[0]
    except Exception:
        logger.debug("Could not resolve project_id for conversation %s", conversation_id)
    return None


async def run_agent_via_hatchet(agent_input: AgentRunInput) -> bool:
    """Run one agent turn on the Hatchet worker. Returns True if completed.

    Mirrors execute_agent_background's orchestration (incl. logfire trace metadata
    so Pydantic AI / LiteLLM spans are queryable by user_id, conversation_id, run_id,
    agent_id, project_id) but async-native; streams to the same Redis stream so
    reconnect/replay and stop behave identically to Celery.
    """
    from app.core.database import SessionLocal, create_celery_async_session
    from app.modules.conversations.conversation.conversation_service import (
        ConversationService,
    )
    from app.modules.conversations.conversation.conversation_store import (
        ConversationStore,
    )
    from app.modules.conversations.message.message_model import MessageType
    from app.modules.conversations.message.message_schema import (
        MessageRequest,
        normalize_node_contexts,
    )
    from app.modules.conversations.message.message_store import MessageStore

    cid = agent_input.conversation_id
    rid = agent_input.run_id
    redis_manager = RedisStreamManager()
    sink = RedisStreamSink(redis_manager, cid, rid)
    operation = getattr(agent_input, "operation", AGENT_RUN_OPERATION_MESSAGE)
    # Hatchet payloads may carry string or dict-shaped node ids; normalize once
    # so message and regenerate paths share the same NodeContext list.
    node_contexts = normalize_node_contexts(agent_input.node_ids)

    # Pre-declare so the finally cleanup is safe even if SessionLocal() /
    # create_celery_async_session() raise before assignment.
    sync_db = None
    async_session = None
    engine = None

    try:
        sync_db = SessionLocal()
        project_id = _resolve_project_id(sync_db, cid)
        user_email = _resolve_user_email(sync_db, agent_input.user_id)
        async_session, engine = create_celery_async_session()

        with logfire_trace_metadata(
            user_id=agent_input.user_id,
            conversation_id=cid,
            run_id=rid,
            agent_id=agent_input.agent_id or "default",
            project_id=project_id,
        ), log_context(
            conversation_id=cid, user_id=agent_input.user_id, run_id=rid
        ):
            logger.info(
                "Hatchet agent run starting (agent_id=%s, operation=%s)",
                agent_input.agent_id,
                operation,
            )
            sink.set_status("running")
            init_usage_context()

            service = ConversationService.create(
                conversation_store=ConversationStore(sync_db, async_session),
                message_store=MessageStore(sync_db, async_session),
                db=sync_db,
                user_id=agent_input.user_id,
                user_email=user_email,
            )

            def check_cancelled() -> bool:
                return redis_manager.check_cancellation(cid, rid)

            def flush_partial():
                message_id = service.history_manager.flush_message_buffer(
                    cid, MessageType.AI_GENERATED
                )
                if message_id:
                    logger.debug("Flushed partial AI response", message_id=message_id)

            if operation == AGENT_RUN_OPERATION_REGENERATE:
                start_payload = {
                    "agent_id": agent_input.agent_id or "default",
                    "status": "processing",
                    "message": "Starting regeneration processing",
                }
                chunk_stream = service.regenerate_last_message_background(
                    cid,
                    node_contexts,
                    agent_input.attachment_ids or [],
                    local_mode=agent_input.local_mode,
                    run_id=rid,
                    check_cancelled=check_cancelled,
                )
                cancel_message = "Regeneration cancelled by user"
                completed_message = "Regeneration completed"
            else:
                message_request = MessageRequest(
                    content=agent_input.query,
                    node_ids=node_contexts,
                    attachment_ids=agent_input.attachment_ids or None,
                    tunnel_url=agent_input.tunnel_url,
                )
                start_payload = {
                    "agent_id": agent_input.agent_id or "default",
                    "status": "processing",
                    "message": "Starting message processing",
                }
                chunk_stream = service.store_message(
                    cid,
                    message_request,
                    MessageType.HUMAN,
                    agent_input.user_id,
                    stream=True,
                    local_mode=agent_input.local_mode,
                    run_id=rid,
                    check_cancelled=check_cancelled,
                )
                cancel_message = "Generation cancelled by user"
                completed_message = "Agent execution completed"

            completed = await run_agent_turn(
                start_payload=start_payload,
                chunk_stream=chunk_stream,
                sink=sink,
                flush_partial=flush_partial,
                cancel_message=cancel_message,
            )

            usages = get_and_clear_usages()
            if completed:
                end_payload = {
                    "status": "completed",
                    "message": completed_message,
                }
                if usages:
                    end_payload["usage_json"] = usages
                sink.emit("end", end_payload)
                sink.set_status("completed")
                logger.info("Hatchet agent run completed")
            else:
                sink.set_status("cancelled")
                logger.info("Hatchet agent run cancelled")
            return completed
    except Exception:
        logger.exception("Hatchet agent run failed")
        try:
            sink.set_status("error")
            sink.emit(
                "end",
                {"status": "error", "message": "An internal error occurred."},
            )
        except Exception:
            logger.exception("Failed to publish error end-event")
        raise
    finally:
        if async_session is not None:
            try:
                await async_session.close()
            except Exception:
                logger.exception("Error closing async DB session")
        if engine is not None:
            try:
                await engine.dispose()
            except Exception:
                logger.exception("Error disposing async DB engine")
        if sync_db is not None:
            try:
                sync_db.close()
            except Exception:
                logger.exception("Error closing sync DB session")


def run_hatchet_agent_worker(hatchet=None) -> None:
    """Register the agent task and block until the worker exits.

    ``hatchet`` is injectable for tests; in production it is built from env.
    """
    if hatchet is None:
        from hatchet_sdk import Hatchet

        prepare_hatchet_client_env()
        hatchet = Hatchet()

    @hatchet.task(
        on_events=[EVENT_AGENT_RUN],
        input_validator=AgentRunInput,
        execution_timeout=timedelta(hours=1),
    )
    async def agent_run_task(agent_input: AgentRunInput, ctx) -> dict:
        del ctx
        completed = await run_agent_via_hatchet(agent_input)
        return {"completed": completed}

    worker = hatchet.worker("potpie-agents", slots=4, workflows=[agent_run_task])
    worker.start()


if __name__ == "__main__":
    run_hatchet_agent_worker()
