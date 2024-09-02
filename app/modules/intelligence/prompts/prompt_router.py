from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional


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
def list_prompts(query: Optional[str] = Query(None, min_length=3), skip: int = 0, limit: int = 10, db=Depends(get_db)):
    return PromptController.list_prompts(query, skip, limit, db)

@router.get("/prompts/all", response_model=PromptListResponse)
def get_all_prompts(skip: int = 0, limit: int = 10, db=Depends(get_db)):
    return PromptController.get_all_prompts(skip, limit, db)
