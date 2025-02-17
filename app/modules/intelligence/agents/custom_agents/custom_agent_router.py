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
            role=request.role,
            goal=request.goal,
            backstory=request.backstory,
            system_prompt=request.system_prompt,
            tasks=[task.model_dump(exclude_unset=True) for task in request.tasks],
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error creating custom agent: {str(e)}")
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
