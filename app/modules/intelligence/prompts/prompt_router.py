from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session


from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.intelligence.prompts.prompt_controller import PromptController
from app.modules.intelligence.prompts.prompt_schema import (
    PromptCreate,
    PromptListResponse,
    PromptResponse,
    PromptUpdate,
)
from app.modules.intelligence.prompts.prompt_service import PromptService
from app.modules.intelligence.prompts.prompt_schema import RequestModel


router = APIRouter()


class PromptAPI:
    @staticmethod
    @router.post("/prompts/", response_model=PromptResponse)
    async def create_prompt(
        prompt: PromptCreate,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        prompt_service = PromptService(db)
        return await PromptController.create_prompt(
            prompt, prompt_service, user["user_id"]
        )

    @staticmethod
    @router.put("/prompts/{prompt_id}", response_model=PromptResponse)
    async def update_prompt(
        prompt_id: str,
        prompt: PromptUpdate,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        prompt_service = PromptService(db)
        return await PromptController.update_prompt(
            prompt_id, prompt, prompt_service, user["user_id"]
        )

    @staticmethod
    @router.delete("/prompts/{prompt_id}", response_model=None)
    async def delete_prompt(
        prompt_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        prompt_service = PromptService(db)
        return await PromptController.delete_prompt(
            prompt_id, prompt_service, user["user_id"]
        )

    @staticmethod
    @router.get("/prompts/{prompt_id}", response_model=PromptResponse)
    async def fetch_prompt(
        prompt_id: str,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        prompt_service = PromptService(db)
        return await PromptController.fetch_prompt(
            prompt_id, prompt_service, user["user_id"]
        )

    @staticmethod
    @router.get("/prompts/", response_model=PromptListResponse)
    async def list_prompts(
        query: Optional[str] = Query(None),
        skip: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=100),
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):
        prompt_service = PromptService(db)
        return await PromptController.list_prompts(
            query, skip, limit, prompt_service, user["user_id"]
        )

    @staticmethod
    @router.post("/prompts/enhancer", response_model=str)
    async def enhance_prompt(
        request_body: RequestModel,
        db: Session = Depends(get_db),
        user=Depends(AuthService.check_auth),
    ):

        return await PromptController.enhance_prompt(request_body, db, user)
