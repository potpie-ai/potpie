from sqlalchemy.orm import Session
from typing import Optional, List
from fastapi import HTTPException
import uuid

from app.modules.intelligence.prompts.prompt_model import Prompt, PromptAccess
from app.modules.intelligence.prompts.prompt_schema import PromptCreate, PromptResponse, PromptUpdate, PromptVisibilityType, PromptStatusType

class PromptService:
    def __init__(self, db: Session):
        self.db = db

    async def create_prompt(self, prompt_data: PromptCreate, user_id: str) -> PromptResponse:
        prompt = Prompt(
            id=str(uuid.uuid4()),
            **prompt_data.dict(),
            type=PromptType.USER,  # Always set to USER type
            version=1,  # Always start with version 1
            created_by=user_id
        )
        self.db.add(prompt)
        self.db.commit()
        self.db.refresh(prompt)
        return PromptResponse.model_validate(prompt)

    async def update_prompt(self, prompt_id: str, prompt_data: PromptUpdate, user_id: str) -> PromptResponse:
        prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        if prompt.created_by != user_id:
            raise HTTPException(status_code=403, detail="You don't have permission to update this prompt")
        
        # Increment version
        prompt.version += 1
        
        for key, value in prompt_data.dict(exclude_unset=True).items():
            setattr(prompt, key, value)
        self.db.commit()
        self.db.refresh(prompt)
        return PromptResponse.model_validate(prompt)

    async def delete_prompt(self, prompt_id: str, user_id: str) -> None:
        prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        if prompt.created_by != user_id:
            raise HTTPException(status_code=403, detail="You don't have permission to delete this prompt")
        self.db.delete(prompt)
        self.db.commit()

    async def fetch_prompt(self, prompt_id: str, user_id: str) -> PromptResponse:
        prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        if prompt.created_by != user_id and prompt.visibility != PromptVisibilityType.PUBLIC:
            raise HTTPException(status_code=403, detail="You don't have permission to view this prompt")
        return PromptResponse.model_validate(prompt)

    async def list_prompts(self, query: Optional[str], skip: int, limit: int, user_id: str) -> List[PromptResponse]:
        base_query = self.db.query(Prompt).filter(
            ((Prompt.created_by == user_id) | (Prompt.visibility == PromptVisibilityType.PUBLIC)) &
            (Prompt.status == PromptStatusType.ACTIVE)
        )
        
        if query:
            base_query = base_query.filter(Prompt.text.ilike(f"%{query}%"))
        
        prompts = base_query.order_by(Prompt.updated_at.desc()).offset(skip).limit(limit).all()
        return [PromptResponse.model_validate(prompt) for prompt in prompts]
    
    async def get_all_prompts(self, skip: int, limit: int, user_id: str) -> List[PromptResponse]:
        # Subquery for user's private prompts
        private_prompts = self.db.query(Prompt)\
            .filter(Prompt.created_by == user_id,
                    Prompt.visibility == PromptVisibilityType.PRIVATE,
                    Prompt.status == PromptStatusType.ACTIVE)\
            .order_by(Prompt.updated_at.desc())

        # Subquery for user's public prompts
        user_public_prompts = self.db.query(Prompt)\
            .filter(Prompt.created_by == user_id,
                    Prompt.visibility == PromptVisibilityType.PUBLIC,
                    Prompt.status == PromptStatusType.ACTIVE)\
            .order_by(Prompt.updated_at.desc())

        # Subquery for public prompts that don't belong to the user
        public_prompts = self.db.query(Prompt)\
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