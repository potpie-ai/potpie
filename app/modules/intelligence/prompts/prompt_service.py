from sqlalchemy.orm import Session
from typing import Optional, List
from fastapi import HTTPException

from .models import Prompt  # Assuming this is your Prompt SQLAlchemy model
from .schemas import PromptCreate, PromptUpdate, PromptResponse

class PromptService:

    @staticmethod
    def create_prompt(db: Session, prompt_data: PromptCreate) -> Prompt:
        prompt = Prompt(**prompt_data.dict())
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
        return prompt

    @staticmethod
    def update_prompt(db: Session, prompt_id: str, prompt_data: PromptUpdate) -> Prompt:
        prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        for key, value in prompt_data.dict(exclude_unset=True).items():
            setattr(prompt, key, value)
        db.commit()
        db.refresh(prompt)
        return prompt

    @staticmethod
    def delete_prompt(db: Session, prompt_id: str) -> None:
        prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        db.delete(prompt)
        db.commit()

    @staticmethod
    def fetch_prompt(db: Session, prompt_id: str) -> Prompt:
        prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        return prompt

    @staticmethod
    def list_prompts(db: Session, query: Optional[str], skip: int, limit: int) -> List[Prompt]:
        query_filter = Prompt.text.ilike(f"%{query}%") if query else True
        return db.query(Prompt).filter(query_filter).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_all_prompts(db: Session, skip: int, limit: int) -> List[Prompt]:
        # First, fetch system prompts
        system_prompts = db.query(Prompt)\
                           .filter(Prompt.type == 'SYSTEM')\
                           .order_by(Prompt.created_at)\
                           .offset(skip)\
                           .limit(limit)\
                           .all()

        # Then, fetch user prompts
        user_prompts = db.query(Prompt)\
                         .filter(Prompt.type == 'USER')\
                         .order_by(Prompt.created_at)\
                         .offset(skip)\
                         .limit(limit)\
                         .all()

        # Combine results: System prompts first, then user prompts
        all_prompts = system_prompts + user_prompts
        return all_prompts