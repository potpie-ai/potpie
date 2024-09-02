from fastapi import APIRouter, Depends, Query
from typing import Optional

from app.core.database import get_db
from app.modules.intelligence.prompts.prompt_controller import PromptController
from app.modules.intelligence.prompts.prompt_schema import PromptCreate, PromptListResponse, PromptResponse, PromptUpdate


router = APIRouter()

@router.post("/prompts", response_model=PromptResponse)
def create_prompt(prompt: PromptCreate, db=Depends(get_db)):
    return PromptController.create_prompt(prompt, db)

@router.put("/prompts/{prompt_id}", response_model=PromptResponse)
def update_prompt(prompt_id: str, prompt: PromptUpdate, db=Depends(get_db)):
    return PromptController.update_prompt(prompt_id, prompt, db)

@router.delete("/prompts/{prompt_id}", response_model=None)
def delete_prompt(prompt_id: str, db=Depends(get_db)):
    return PromptController.delete_prompt(prompt_id, db)

@router.get("/prompts/{prompt_id}", response_model=PromptResponse)
def fetch_prompt(prompt_id: str, db=Depends(get_db)):
    return PromptController.fetch_prompt(prompt_id, db)

@router.get("/prompts", response_model=PromptListResponse)
def list_prompts(
    query: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db=Depends(get_db)
):
    if query and len(query) >= 5:
        return PromptController.list_prompts(query, skip, limit, db)
    else:
        return PromptController.get_all_prompts(skip, limit, db)
