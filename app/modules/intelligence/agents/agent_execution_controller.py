import json
from typing import Optional
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.conversations.utils.redis_streaming import RedisStreamManager
from app.modules.intelligence.agents.agent_execution_schema import (
    AgentExecuteRequest,
    AgentExecuteStartResponse,
    AgentExecutionResultResponse,
)
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.projects.projects_service import ProjectService
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)

EXECUTION_RESULT_PREFIX = "agent_execution_result:"
EXECUTION_STATUS_PREFIX = "agent_execution_status:"
EXECUTION_TTL = 3600


class AgentExecutionController:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.redis_manager = RedisStreamManager()
        self.project_service = ProjectService(db)

        provider_service = ProviderService(db, user_id)
        prompt_service = PromptService(db)
        tool_service = ToolService(db, user_id)

        self.agent_service = AgentsService(
            db, provider_service, prompt_service, tool_service
        )

    async def start_execution(
        self, agent_id: str, request: AgentExecuteRequest
    ) -> AgentExecuteStartResponse:
        """Validate agent and project, dispatch Celery task, return execution_id"""

        agent_type = await self.agent_service.validate_agent_id(self.user_id, agent_id)
        if agent_type is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        project = await self.project_service.get_project_from_db_by_id(
            request.project_id
        )
        if project is None:
            raise HTTPException(
                status_code=404, detail=f"Project '{request.project_id}' not found"
            )

        execution_id = str(uuid4())

        self._set_execution_status(execution_id, "queued")

        from app.celery.tasks.agent_execution_task import execute_agent_direct

        task = execute_agent_direct.delay(
            execution_id=execution_id,
            user_id=self.user_id,
            agent_id=agent_id,
            agent_type=agent_type,
            project_id=request.project_id,
            project_name=project.get("project_name") or project.get("id"),
            query=request.query,
            node_ids=request.node_ids,
            additional_context=request.additional_context or "",
        )

        self.redis_manager.redis_client.set(
            f"agent_execution_task:{execution_id}",
            task.id,
            ex=EXECUTION_TTL,
        )

        logger.info(
            "Started agent execution",
            execution_id=execution_id,
            agent_id=agent_id,
            agent_type=agent_type,
            task_id=task.id,
        )

        return AgentExecuteStartResponse(execution_id=execution_id, status="queued")

    def get_execution_result(self, execution_id: str) -> AgentExecutionResultResponse:
        """Get the status and result of an agent execution"""

        status = self._get_execution_status(execution_id)
        if status is None:
            raise HTTPException(
                status_code=404, detail=f"Execution '{execution_id}' not found"
            )

        if status in ("queued", "running"):
            return AgentExecutionResultResponse(
                execution_id=execution_id,
                status=status,
                response=None,
                citations=None,
                error=None,
            )

        result_data = self._get_execution_result(execution_id)
        if result_data is None:
            if status == "failed":
                return AgentExecutionResultResponse(
                    execution_id=execution_id,
                    status="failed",
                    response=None,
                    citations=None,
                    error="Execution failed without result data",
                )
            return AgentExecutionResultResponse(
                execution_id=execution_id,
                status=status,
                response=None,
                citations=None,
                error=None,
            )

        return AgentExecutionResultResponse(
            execution_id=execution_id,
            status=status,
            response=result_data.get("response"),
            citations=result_data.get("citations"),
            error=result_data.get("error"),
        )

    def _set_execution_status(self, execution_id: str, status: str):
        """Store execution status in Redis"""
        self.redis_manager.redis_client.set(
            f"{EXECUTION_STATUS_PREFIX}{execution_id}",
            status,
            ex=EXECUTION_TTL,
        )

    def _get_execution_status(self, execution_id: str) -> Optional[str]:
        """Get execution status from Redis"""
        status = self.redis_manager.redis_client.get(
            f"{EXECUTION_STATUS_PREFIX}{execution_id}"
        )
        if status:
            return status.decode("utf-8") if isinstance(status, bytes) else status
        return None

    def _get_execution_result(self, execution_id: str) -> Optional[dict]:
        """Get execution result from Redis"""
        result = self.redis_manager.redis_client.get(
            f"{EXECUTION_RESULT_PREFIX}{execution_id}"
        )
        if result:
            result_str = result.decode("utf-8") if isinstance(result, bytes) else result
            return json.loads(result_str)
        return None
