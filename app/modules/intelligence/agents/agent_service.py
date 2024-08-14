from sqlalchemy.orm import Session

from app.modules.intelligence.agents.agents_model import Agent

class AgentService:
    def __init__(self, db: Session):
        self.db = db

    def get_agents(self) -> list[Agent]:
        return self.db.query(Agent).all()
