from typing import Optional

from fastapi import HTTPException

from app.modules.intelligence.prompts.prompt_schema import (
    PromptCreate,
    PromptListResponse,
    PromptResponse,
    PromptUpdate,
)
from app.modules.intelligence.prompts.prompt_service import (
    PromptNotFoundError,
    PromptService,
    PromptServiceError,
)


class PromptController:
    @staticmethod
    async def create_prompt(
        prompt: PromptCreate, prompt_service: PromptService, user_id: str
    ) -> PromptResponse:
        try:
            return await prompt_service.create_prompt(prompt, user_id)
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def update_prompt(
        prompt_id: str,
        prompt: PromptUpdate,
        prompt_service: PromptService,
        user_id: str,
    ) -> PromptResponse:
        try:
            return await prompt_service.update_prompt(prompt_id, prompt, user_id)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def delete_prompt(
        prompt_id: str, prompt_service: PromptService, user_id: str
    ) -> dict:
        try:
            await prompt_service.delete_prompt(prompt_id, user_id)
            return {"message": "Prompt deleted successfully"}
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def fetch_prompt(
        prompt_id: str, prompt_service: PromptService, user_id: str
    ) -> PromptResponse:
        try:
            return await prompt_service.fetch_prompt(prompt_id, user_id)
        except PromptNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def list_prompts(
        query: Optional[str],
        skip: int,
        limit: int,
        prompt_service: PromptService,
        user_id: str,
    ) -> PromptListResponse:
        try:
            return await prompt_service.list_prompts(query, skip, limit, user_id)
        except PromptServiceError as e:
            raise HTTPException(status_code=500, detail=str(e))
