from typing import Optional
from fastapi import HTTPException
from app.modules.intelligence.prompts.prompt_schema import PromptCreate, PromptListResponse, PromptResponse, PromptUpdate
from app.modules.intelligence.prompts.prompt_service import PromptService


class PromptController:

    @staticmethod
    async def create_prompt(prompt: PromptCreate, prompt_service: PromptService, user_id: str) -> PromptResponse:
        try:
            return await prompt_service.create_prompt(prompt, user_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to create prompt: {str(e)}")

    @staticmethod
    async def update_prompt(prompt_id: str, prompt: PromptUpdate, prompt_service: PromptService, user_id: str) -> PromptResponse:
        try:
            return await prompt_service.update_prompt(prompt_id, prompt, user_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to update prompt: {str(e)}")

    @staticmethod
    async def delete_prompt(prompt_id: str, prompt_service: PromptService, user_id: str) -> None:
        try:
            await prompt_service.delete_prompt(prompt_id, user_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to delete prompt: {str(e)}")

    @staticmethod
    async def fetch_prompt(prompt_id: str, prompt_service: PromptService, user_id: str) -> PromptResponse:
        try:
            return await prompt_service.fetch_prompt(prompt_id, user_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Prompt not found: {str(e)}")

    @staticmethod
    async def list_prompts(query: Optional[str], skip: int, limit: int, prompt_service: PromptService, user_id: str) -> PromptListResponse:
        try:
            prompts = await prompt_service.list_prompts(query, skip, limit, user_id)
            return PromptListResponse(prompts=prompts, total=len(prompts))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to list prompts: {str(e)}")

    @staticmethod
    async def get_all_prompts(skip: int, limit: int, prompt_service: PromptService, user_id: str) -> PromptListResponse:
        try:
            prompts = await prompt_service.get_all_prompts(skip, limit, user_id)
            return PromptListResponse(prompts=prompts, total=len(prompts))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to get all prompts: {str(e)}")