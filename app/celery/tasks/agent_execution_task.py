import json
from typing import List, Optional

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.utils.logger import setup_logger, log_context

logger = setup_logger(__name__)

EXECUTION_RESULT_PREFIX = "agent_execution_result:"
EXECUTION_STATUS_PREFIX = "agent_execution_status:"
EXECUTION_TTL = 3600


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.agent_execution_task.execute_agent_direct",
)
def execute_agent_direct(
    self,
    execution_id: str,
    user_id: str,
    agent_id: str,
    agent_type: str,
    project_id: str,
    project_name: str,
    query: str,
    node_ids: Optional[List[str]] = None,
    additional_context: str = "",
) -> bool:
    """Execute an agent directly and store the full result in Redis"""
    from app.modules.conversations.utils.redis_streaming import RedisStreamManager

    redis_manager = RedisStreamManager()

    with log_context(execution_id=execution_id, user_id=user_id, agent_id=agent_id):
        logger.info("Starting direct agent execution")

        _set_execution_status(redis_manager, execution_id, "running")

        try:

            async def run_agent():
                from app.modules.intelligence.agents.agents_service import AgentsService
                from app.modules.intelligence.agents.chat_agent import ChatContext
                from app.modules.intelligence.provider.provider_service import (
                    ProviderService,
                )
                from app.modules.intelligence.prompts.prompt_service import (
                    PromptService,
                )
                from app.modules.intelligence.tools.tool_service import ToolService

                provider_service = ProviderService(self.db, user_id)
                prompt_service = PromptService(self.db)
                tool_service = ToolService(self.db, user_id)

                agent_service = AgentsService(
                    self.db, provider_service, prompt_service, tool_service
                )

                ctx = ChatContext(
                    project_id=project_id,
                    project_name=project_name,
                    curr_agent_id=agent_id,
                    history=[],
                    node_ids=node_ids or [],
                    additional_context=additional_context,
                    query=query,
                )

                full_response = ""
                all_citations = []

                if agent_type == "CUSTOM_AGENT":
                    res = (
                        await agent_service.custom_agent_service.execute_agent_runtime(
                            user_id, ctx
                        )
                    )
                    async for chunk in res:
                        full_response += chunk.response
                        all_citations.extend(chunk.citations)
                else:
                    res = agent_service.execute_stream(ctx)
                    async for chunk in res:
                        full_response += chunk.response
                        all_citations.extend(chunk.citations)

                unique_citations = list(dict.fromkeys(all_citations))

                return {
                    "response": full_response,
                    "citations": unique_citations,
                    "error": None,
                }

            result = self.run_async(run_agent())

            _store_execution_result(redis_manager, execution_id, result)
            _set_execution_status(redis_manager, execution_id, "completed")

            logger.info(
                "Direct agent execution completed",
                execution_id=execution_id,
                response_length=len(result.get("response", "")),
            )

            return True

        except Exception as e:
            logger.exception(
                "Direct agent execution failed",
                execution_id=execution_id,
                user_id=user_id,
                agent_id=agent_id,
            )

            error_result = {
                "response": None,
                "citations": None,
                "error": str(e),
            }
            _store_execution_result(redis_manager, execution_id, error_result)
            _set_execution_status(redis_manager, execution_id, "failed")

            raise


def _set_execution_status(redis_manager, execution_id: str, status: str):
    """Store execution status in Redis"""
    redis_manager.redis_client.set(
        f"{EXECUTION_STATUS_PREFIX}{execution_id}",
        status,
        ex=EXECUTION_TTL,
    )


def _store_execution_result(redis_manager, execution_id: str, result: dict):
    """Store execution result in Redis"""
    redis_manager.redis_client.set(
        f"{EXECUTION_RESULT_PREFIX}{execution_id}",
        json.dumps(result),
        ex=EXECUTION_TTL,
    )
