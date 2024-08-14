from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.modules.intelligence.agents.agent_schema import AgentResponse
from app.modules.intelligence.agents.agent_service import AgentService


class AgentController:
    def __init__(self, db: Session = Depends(get_db)):
        self.agent_service = AgentService(db)

    async def list_agents(self) -> list[AgentResponse]:
        agents = self.agent_service.get_agents()
        return [AgentResponse.model_validate(agent) for agent in agents]
