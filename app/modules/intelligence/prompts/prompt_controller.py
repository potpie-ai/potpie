from sqlalchemy.orm import Session
from fastapi import HTTPException

from .prompt_service import PromptService
from .schemas import PromptCreate, PromptUpdate, PromptResponse, PromptListResponse

class PromptController:

    @staticmethod
    def create_prompt(prompt_data: PromptCreate, db: Session) -> PromptResponse:
        prompt = PromptService.create_prompt(db, prompt_data)
        return PromptResponse.from_orm(prompt)

    @staticmethod
    def update_prompt(prompt_id: str, prompt_data: PromptUpdate, db: Session) -> PromptResponse:
        prompt = PromptService.update_prompt(db, prompt_id, prompt_data)
        return PromptResponse.from_orm(prompt)

    @staticmethod
    def delete_prompt(prompt_id: str, db: Session) -> None:
        PromptService.delete_prompt(db, prompt_id)

    @staticmethod
    def fetch_prompt(prompt_id: str, db: Session) -> PromptResponse:
        prompt = PromptService.fetch_prompt(db, prompt_id)
        return PromptResponse.from_orm(prompt)

    @staticmethod
    def list_prompts(query: Optional[str], skip: int, limit: int, db: Session) -> PromptListResponse:
        prompts = PromptService.list_prompts(db, query, skip, limit)
        return PromptListResponse(prompts=prompts, total=len(prompts))

    @staticmethod
    def get_all_prompts(skip: int, limit: int, db: Session) -> PromptListResponse:
        prompts = PromptService.get_all_prompts(db, skip, limit)
        return PromptListResponse(prompts=prompts, total=len(prompts))