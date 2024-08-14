from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.modules.intelligence.agents.agents_model import Agent, AgentStatus
from uuid6 import uuid7

class AgentsSetup:
    def setup_agents(self):
        agents = [
            {"name": "langchain_duckduckgo", "provider": "LANGCHAIN", "description": "A LangChain agent for duckduckgo search"},
        ]

        db: Session = SessionLocal()
        try:
            for agent in agents:
                existing_agent = db.query(Agent).filter_by(name=agent["name"]).first()
                if not existing_agent:
                    new_agent = Agent(
                        id=str(uuid7()),  # Assuming you're generating UUIDs
                        name=agent["name"],
                        provider=agent["provider"],
                        description=agent.get("description"),
                        status=AgentStatus.ACTIVE
                    )
                    db.add(new_agent)
                    db.commit()
                    print(f"Added agent: {agent['name']}")
                else:
                    print(f"Agent {agent['name']} already exists")
        finally:
            db.close()
