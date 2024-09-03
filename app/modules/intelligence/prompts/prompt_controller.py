from typing import Optional
from fastapi import HTTPException
from app.modules.intelligence.prompts.prompt_schema import PromptCreate, PromptListResponse, PromptResponse, PromptUpdate
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.prompts.system_prompt_setup import SystemPromptSetup
from sqlalchemy.orm import Session


class PromptController:

    def __init__(self, db: Session):
        self.prompt_service = PromptService(db)
        self.system_prompt_setup = SystemPromptSetup(db)

    @staticmethod
    async def create_prompt(prompt: PromptCreate, user_id: str):
        return await self.prompt_service.create_prompt(prompt, user_id)

    @staticmethod
    async def update_prompt(prompt_id: str, prompt: PromptUpdate, user_id: str):
        return await self.prompt_service.update_prompt(prompt_id, prompt, user_id)

    @staticmethod
    async def delete_prompt(prompt_id: str, user_id: str):
        await self.prompt_service.delete_prompt(prompt_id, user_id)

    @staticmethod
    async def fetch_prompt(prompt_id: str, user_id: str):
        return await self.prompt_service.fetch_prompt(prompt_id, user_id)

    @staticmethod
    async def list_prompts(query: Optional[str], skip: int, limit: int, user_id: str):
        return await self.prompt_service.list_prompts(query, skip, limit, user_id)

    async def initialize_system_prompts(self):
        await self.system_prompt_setup.initialize_system_prompts()

    async def get_system_prompts(self, agent_id: str):
        return await self.system_prompt_setup.get_system_prompts(agent_id)