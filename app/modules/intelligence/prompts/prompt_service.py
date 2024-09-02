from sqlalchemy.orm import Session
from typing import Optional, List
from fastapi import HTTPException

from app.modules.intelligence.prompts.prompt_model import Prompt
from app.modules.intelligence.prompts.prompt_schema import PromptCreate, PromptResponse, PromptUpdate, PromptVisibilityType, PromptStatusType

class PromptService:

    @staticmethod
    def create_prompt(db: Session, prompt_data: PromptCreate) -> PromptResponse:
        prompt = Prompt(**prompt_data.dict())
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
        return PromptResponse.model_validate(prompt)

    @staticmethod
    def update_prompt(db: Session, prompt_id: str, prompt_data: PromptUpdate) -> PromptResponse:
        prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        for key, value in prompt_data.dict(exclude_unset=True).items():
            setattr(prompt, key, value)
        db.commit()
        db.refresh(prompt)
        return PromptResponse.model_validate(prompt)

    @staticmethod
    def delete_prompt(db: Session, prompt_id: str) -> None:
        prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        db.delete(prompt)
        db.commit()

    @staticmethod
    def fetch_prompt(db: Session, prompt_id: str) -> PromptResponse:
        prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return PromptResponse.model_validate(prompt)

    @staticmethod
    def list_prompts(db: Session, query: Optional[str], skip: int, limit: int) -> List[PromptResponse]:
        query_filter = Prompt.text.ilike(f"%{query}%") if query else True
        prompts = db.query(Prompt).filter(query_filter).offset(skip).limit(limit).all()
        return [PromptResponse.model_validate(prompt) for prompt in prompts]
    
    @staticmethod
    def get_all_prompts(db: Session, skip: int, limit: int, user_id: str) -> List[PromptResponse]:
        # Subquery for user's private prompts
        private_prompts = db.query(Prompt)\
            .filter(Prompt.created_by == user_id,
                    Prompt.visibility == PromptVisibilityType.PRIVATE,
                    Prompt.status == PromptStatusType.ACTIVE)\
            .order_by(Prompt.updated_at.desc())

        # Subquery for user's public prompts
        user_public_prompts = db.query(Prompt)\
            .filter(Prompt.created_by == user_id,
                    Prompt.visibility == PromptVisibilityType.PUBLIC,
                    Prompt.status == PromptStatusType.ACTIVE)\
            .order_by(Prompt.updated_at.desc())

        # Subquery for public prompts that don't belong to the user
        public_prompts = db.query(Prompt)\
            .filter(Prompt.created_by != user_id,
                    Prompt.visibility == PromptVisibilityType.PUBLIC,
                    Prompt.status == PromptStatusType.ACTIVE)\
            .order_by(Prompt.updated_at.desc())

        # Combine all prompts using union_all
        combined_query = private_prompts.union_all(user_public_prompts, public_prompts)\
            .order_by(Prompt.updated_at.desc())

        # Apply skip and limit to the combined query
        paginated_prompts = combined_query.offset(skip).limit(limit).all()

        return [PromptResponse.model_validate(prompt) for prompt in paginated_prompts]