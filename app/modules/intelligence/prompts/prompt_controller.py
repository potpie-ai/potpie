from typing import Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.modules.intelligence.prompts.prompt_schema import (
    PromptCreate,
    PromptListResponse,
    PromptResponse,
    PromptUpdate,
)
from app.modules.intelligence.prompts.prompt_service import (
    PromptNotFoundError,
    PromptService,
    PromptServiceError,
)
from app.modules.intelligence.agents.agents_service import AgentsService
from app.modules.intelligence.prompts.prompt_schema import RequestModel
from app.modules.conversations.conversation.conversation_model import Conversation
from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent
from app.modules.conversations.message.message_model import (
    Message,
    MessageStatus,
)
from app.modules.conversations.message.message_schema import MessageResponse
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService


class PromptController:
    @staticmethod
    async def create_prompt(
        prompt: PromptCreate, prompt_service: PromptService, user_id: str
    ) -> PromptResponse:
        try:
            return await prompt_service.create_prompt(prompt, user_id)
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def update_prompt(
        prompt_id: str,
        prompt: PromptUpdate,
        prompt_service: PromptService,
        user_id: str,
    ) -> PromptResponse:
        try:
            return await prompt_service.update_prompt(prompt_id, prompt, user_id)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def delete_prompt(
        prompt_id: str, prompt_service: PromptService, user_id: str
    ) -> dict:
        try:
            await prompt_service.delete_prompt(prompt_id, user_id)
            return {"message": "Prompt deleted successfully"}
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def fetch_prompt(
        prompt_id: str, prompt_service: PromptService, user_id: str
    ) -> PromptResponse:
        try:
            return await prompt_service.fetch_prompt(prompt_id, user_id)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def list_prompts(
        query: Optional[str],
        skip: int,
        limit: int,
        prompt_service: PromptService,
        user_id: str,
    ) -> PromptListResponse:
        try:
            return await prompt_service.list_prompts(query, skip, limit, user_id)
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def enhance_prompt(
        request_body: RequestModel,
        db: Session,
        user: dict,
    ) -> str:

        messages = (
            db.query(Message)
            .filter_by(conversation_id=request_body.conversation_id)
            .filter_by(status=MessageStatus.ACTIVE)
            .order_by(Message.created_at.desc())
            .limit(2)
            .all()
        )

        # to get them in chronological order
        messages.reverse()

        last_messages = [
            MessageResponse(
                id=message.id,
                conversation_id=message.conversation_id,
                content=message.content,
                sender_id=message.sender_id,
                type=message.type,
                status=message.status,
                created_at=message.created_at,
                citations=(message.citations.split(",") if message.citations else None),
            )
            for message in messages
        ]

        llm_provider = ProviderService(db, user["user_id"])
        prompt_provider = PromptService(db)
        tools_provider = ToolService(db, user["user_id"])

        agents_service = AgentsService(
            db, llm_provider, prompt_provider, tools_provider
        )
        available_agents = await agents_service.list_available_agents(
            user, list_system_agents=True
        )

        agent_ids = (
            db.query(Conversation.agent_ids)
            .filter_by(id=request_body.conversation_id)
            .scalar()
        ) or []

        is_custom_agent = (
            db.query(CustomAgent).filter(CustomAgent.id.in_(agent_ids)).first()
            is not None
        )

        if not is_custom_agent:
            enhanced_prompt = await PromptService(db).enhance_prompt(
                request_body.prompt, last_messages, user, agent_ids, available_agents
            )
        else:
            enhanced_prompt = await PromptService(db).enhance_prompt(
                request_body.prompt,
                last_messages,
                user,
            )
        return enhanced_prompt
