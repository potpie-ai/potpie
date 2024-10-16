import os
from typing import Dict, Any, AsyncGenerator
import aiohttp
from langchain.llms import BaseLLM

class CustomAgent:
    def __init__(self, llm: BaseLLM, agent_id: str):
        self.llm = llm
        self.agent_id = agent_id
        self.base_url = os.getenv("CUSTOM_AGENT_BASE_URL")

    async def run(self, query: str, project_id: str, user_id: str, conversation_id: str, node_ids: list) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/run"
        
        params = {
            "agent_id": self.agent_id
        }

        payload = {
            "query": query,
            "project_id": project_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "node_ids": node_ids
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, params=params, json=payload) as response:
                if response.status != 200:
                    error_message = await response.text()
                    raise Exception(f"Custom agent execution failed: {error_message}")

                async for chunk in response.content.iter_any():
                    yield chunk.decode('utf-8')

    async def get_system_prompt(self) -> str:
        url = f"{self.base_url}/system_prompt"
        
        params = {
            "agent_id": self.agent_id
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_message = await response.text()
                    raise Exception(f"Failed to get system prompt: {error_message}")
                
                return await response.text()
