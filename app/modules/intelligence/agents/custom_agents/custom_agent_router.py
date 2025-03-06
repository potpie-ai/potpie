from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import auth_handler
from app.modules.intelligence.agents.custom_agents.custom_agent_controller import (
    CustomAgentController,
)
from app.modules.intelligence.agents.custom_agents.custom_agent_schema import (
    Agent,
    AgentCreate,
    AgentUpdate,
    PromptBasedAgentRequest,
    AgentSharingRequest,
    ListAgentsRequest,
    RevokeAgentAccessRequest,
    AgentSharesResponse,
)
from app.modules.utils.APIRouter import APIRouter
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


@router.post("/", response_model=Agent)
async def create_custom_agent(
    request: AgentCreate,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Create a new custom agent"""
    user_id = user["user_id"]
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.create_agent(
            user_id=user_id,
            agent_data=request,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error creating custom agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/share", response_model=Agent)
async def share_agent(
    request: AgentSharingRequest,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Share an agent with another user or change its visibility"""
    user_id = user["user_id"]
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.manage_agent_sharing(
            agent_id=request.agent_id,
            owner_id=user_id,
            visibility=request.visibility,
            shared_with_email=request.shared_with_email,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error sharing agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/revoke-access", response_model=Agent)
async def revoke_agent_access(
    request: RevokeAgentAccessRequest,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Revoke a specific user's access to an agent"""
    user_id = user["user_id"]
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.revoke_agent_access(
            agent_id=request.agent_id,
            owner_id=user_id,
            user_email=request.user_email,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error revoking agent access: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/", response_model=list[Agent])
async def list_agents(
    request: ListAgentsRequest = Depends(),
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """List all agents accessible to the user including public and shared agents"""
    user_id = user["user_id"]
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.list_agents(
            user_id=user_id,
            include_public=request.include_public,
            include_shared=request.include_shared,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.delete("/{agent_id}", response_model=dict)
async def delete_custom_agent(
    agent_id: str,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Delete a custom agent"""
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.delete_agent(agent_id, user["user_id"])
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
    except Exception as e:
        logger.error(f"Error deleting custom agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.put("/{agent_id}", response_model=Agent)
async def update_custom_agent(
    agent_id: str,
    request: AgentUpdate,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Update a custom agent"""
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.update_agent(
            agent_id, user["user_id"], request
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error updating custom agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/{agent_id}", response_model=Agent)
async def get_custom_agent_info(
    agent_id: str,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Get information about a specific custom agent"""
    logger.info(f"Getting custom agent info for agent_id: {agent_id}")
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.get_agent(agent_id, user["user_id"])
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
    except Exception as e:
        logger.error(f"Error retrieving custom agent info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/{agent_id}/shares", response_model=AgentSharesResponse)
async def get_agent_shares(
    agent_id: str,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Get a list of all emails this agent has been shared with"""
    logger.info(f"Getting shares for agent_id: {agent_id}")
    custom_agent_controller = CustomAgentController(db)
    try:
        shared_emails = await custom_agent_controller.list_agent_shares(
            agent_id, user["user_id"]
        )
        return AgentSharesResponse(agent_id=agent_id, shared_with=shared_emails)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error retrieving agent shares: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/auto/", response_model=Agent)
async def create_agent_from_prompt(
    request: PromptBasedAgentRequest,
    db: Session = Depends(get_db),
    user=Depends(auth_handler.check_auth),
):
    """Create a custom agent from a natural language prompt"""
    custom_agent_controller = CustomAgentController(db)
    try:
        return await custom_agent_controller.create_agent_from_prompt(
            prompt=request.prompt,
            user_id=user["user_id"],
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error creating agent from prompt: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
