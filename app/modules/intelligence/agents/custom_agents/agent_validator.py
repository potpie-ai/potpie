from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.modules.intelligence.agents.custom_agents.custom_agent_model import CustomAgent
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


async def validate_agent(db: Session, user_id: str, agent_id: str) -> bool:
    """Validate if an agent exists and belongs to the user"""
    try:
        agent = (
            db.query(CustomAgent)
            .filter(CustomAgent.id == agent_id, CustomAgent.user_id == user_id)
            .first()
        )
        return agent is not None
    except SQLAlchemyError as e:
        logger.error(f"Error validating agent {agent_id}: {str(e)}")
        return False
