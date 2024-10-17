import httpx
from sqlalchemy.orm import Session

class CustomAgentService:
    def __init__(self, db: Session):
        self.db = db
        self.base_url = "http://localhost:8080"

    async def execute_custom_agent(self, agent_id: str, query: str) -> str:
        deployment_url = f"{self.base_url}/deployment/{agent_id}/query"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                deployment_url,
                json={"query": query}
            )
            response.raise_for_status()
            return response.text