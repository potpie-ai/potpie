from datetime import datetime, timedelta
from typing import List, Optional
import os
import httpx

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.api_key_service import APIKeyService
from app.modules.conversations.conversation.conversation_controller import (
    ConversationController,
)
from app.modules.conversations.conversation.conversation_schema import (
    ConversationStatus,
    CreateConversationRequest,
    CreateConversationResponse,
)
from app.modules.conversations.message.message_schema import (
    DirectMessageRequest,
    MessageRequest,
)
from app.modules.intelligence.agents.agents_controller import AgentsController
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.provider.provider_service import ProviderService
from app.modules.intelligence.tools.tool_service import ToolService
from app.modules.parsing.graph_construction.parsing_controller import ParsingController
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.projects.projects_controller import ProjectController
from app.modules.utils.APIRouter import APIRouter
from app.modules.usage.usage_service import UsageService


router = APIRouter()


class SimpleConversationRequest(BaseModel):
    project_ids: List[str]
    agent_ids: List[str]


async def get_api_key_user(
    x_api_key: Optional[str] = Header(None), db: Session = Depends(get_db)
) -> dict:
    """Dependency to validate API key and get user info."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    user = await APIKeyService.validate_api_key(x_api_key, db)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return user


@router.post("/conversations/", response_model=CreateConversationResponse)
async def create_conversation(
    conversation: SimpleConversationRequest,
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    user_id = user["user_id"]
    subscription_url = os.getenv("SUBSCRIPTION_BASE_URL")
    if subscription_url:
        subscription_url = f"{os.getenv('SUBSCRIPTION_BASE_URL')}/subscriptions/info"
        async with httpx.AsyncClient() as client:
            response = await client.get(subscription_url, params={"user_id": user_id})
            subscription_data = response.json()

        end_date_str = subscription_data.get("end_date")
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        else:
            end_date = datetime.utcnow() 

        start_date = end_date - timedelta(days=30)

        usage_data = await UsageService.get_usage_data(
            start_date=start_date, end_date=end_date, user_id=user_id
        )
        total_human_messages = usage_data["total_human_messages"]

        plan_type = subscription_data.get("plan_type", "free")
        message_limit = 500 if plan_type == "pro" else 50

        if total_human_messages >= message_limit:
            raise HTTPException(
                status_code=402,
                detail=f"Message limit of {message_limit} reached for {plan_type} plan."
            )

    # Create full conversation request with defaults
    full_request = CreateConversationRequest(
        user_id=user_id,
        title=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status=ConversationStatus.ARCHIVED,
        project_ids=conversation.project_ids,
        agent_ids=conversation.agent_ids,
    )

    controller = ConversationController(db, user_id, None)
    return await controller.create_conversation(full_request)


@router.post("/parse")
async def parse_directory(
    repo_details: ParsingRequest,
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    return await ParsingController.parse_directory(repo_details, db, user)


@router.get("/parsing-status/{project_id}")
async def get_parsing_status(
    project_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    return await ParsingController.fetch_parsing_status(project_id, db, user)


@router.post("/conversations/{conversation_id}/message/")
async def post_message(
    conversation_id: str,
    message: MessageRequest,
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    if message.content == "" or message.content is None or message.content.isspace():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    user_id = user["user_id"]
    subscription_url = os.getenv("SUBSCRIPTION_BASE_URL")
    if subscription_url:
        subscription_url = f"{os.getenv('SUBSCRIPTION_BASE_URL')}/subscriptions/info"
        async with httpx.AsyncClient() as client:
            response = await client.get(subscription_url, params={"user_id": user_id})
            subscription_data = response.json()

        end_date_str = subscription_data.get("end_date")
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str)
        else:
            end_date = datetime.utcnow() 

        start_date = end_date - timedelta(days=30)

        usage_data = await UsageService.get_usage_data(
            start_date=start_date, end_date=end_date, user_id=user_id
        )
        total_human_messages = usage_data["total_human_messages"]

        plan_type = subscription_data.get("plan_type", "free")
        message_limit = 500 if plan_type == "pro" else 50

        if total_human_messages >= message_limit:
            raise HTTPException(
                status_code=402,
                detail=f"Message limit of {message_limit} reached for {plan_type} plan."
            )

    # Note: email is no longer available with API key auth
    controller = ConversationController(db, user_id, None)
    message_stream = controller.post_message(conversation_id, message, stream=False)
    async for chunk in message_stream:
        return chunk


@router.post("/project/{project_id}/message/")
async def create_conversation_and_message(
    project_id: str,
    message: DirectMessageRequest,
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    if message.content == "" or message.content is None or message.content.isspace():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    user_id = user["user_id"]

    # default agent_id to codebase_qna_agent
    if message.agent_id is None:
        message.agent_id = "codebase_qna_agent"

    controller = ConversationController(db, user_id, None)
    res = await controller.create_conversation(
        CreateConversationRequest(
            user_id=user_id,
            title=message.content,
            project_ids=[project_id],
            agent_ids=[message.agent_id],
            status=ConversationStatus.ACTIVE,
        )
    )

    message_stream = controller.post_message(
        conversation_id=res.conversation_id,
        message=MessageRequest(content=message.content, node_ids=message.node_ids),
        stream=False,
    )

    async for chunk in message_stream:
        return chunk


@router.get("/projects/list")
async def list_projects(
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    return await ProjectController.get_project_list(user, db)


@router.get("/list-available-agents")
async def list_agents(
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    user_id: str = user["user_id"]
    llm_provider = ProviderService(db, user_id)
    tools_provider = ToolService(db, user_id)
    prompt_provider = PromptService(db)
    controller = AgentsController(db, llm_provider, prompt_provider, tools_provider)
    return await controller.list_available_agents(user, True)
