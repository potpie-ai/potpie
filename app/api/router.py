from datetime import datetime
import os
from typing import List, Optional

from fastapi import Depends, Header, HTTPException, Query
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
from app.modules.users.user_service import UserService
from app.modules.utils.APIRouter import APIRouter
from app.modules.usage.usage_service import UsageService
from app.modules.search.search_service import SearchService
from app.modules.search.search_schema import SearchRequest, SearchResponse

router = APIRouter()


class SimpleConversationRequest(BaseModel):
    project_ids: List[str]
    agent_ids: List[str]


async def get_api_key_user(
    x_api_key: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> dict:
    """Dependency to validate API key and get user info."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key == os.environ.get("INTERNAL_ADMIN_SECRET"):
        user = UserService(db).get_user_by_uid(x_user_id or "")
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid user_id",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return {"user_id": user.uid, "email": user.email, "auth_type": "api_key"}

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
    hidden: bool = Query(
        True, description="Whether to hide this conversation from the web UI"
    ),
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    user_id = user["user_id"]
    # This will either return True or raise an HTTPException
    await UsageService.check_usage_limit(user_id)
    # Create full conversation request with defaults
    full_request = CreateConversationRequest(
        user_id=user_id,
        title=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status=ConversationStatus.ACTIVE,  # Let hidden parameter control the final status
        project_ids=conversation.project_ids,
        agent_ids=conversation.agent_ids,
    )

    controller = ConversationController(db, user_id, None)
    return await controller.create_conversation(full_request, hidden)


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
    checked = await UsageService.check_usage_limit(user_id)
    if not checked:
        raise HTTPException(
            status_code=402,
            detail="Subscription required to create a conversation.",
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
    hidden: bool = Query(
        True, description="Whether to hide this conversation from the web UI"
    ),
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

    # Create conversation with hidden parameter
    res = await controller.create_conversation(
        CreateConversationRequest(
            user_id=user_id,
            title=message.content,
            project_ids=[project_id],
            agent_ids=[message.agent_id],
            status=ConversationStatus.ACTIVE,  # Let hidden parameter control the final status
        ),
        hidden,
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


@router.post("/search", response_model=SearchResponse)
async def search_codebase(
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    user=Depends(get_api_key_user),
):
    """Search codebase using API key authentication"""
    search_service = SearchService(db)
    results = await search_service.search_codebase(
        search_request.project_id, search_request.query
    )
    return SearchResponse(results=results)
