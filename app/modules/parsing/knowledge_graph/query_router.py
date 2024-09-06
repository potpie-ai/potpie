from fastapi import APIRouter, Depends
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.modules.parsing.knowledge_graph.inference_service import InferenceService

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    node_ids: Optional[List[str]] = None

class QueryResponse(BaseModel):
    node_id: str
    docstring: str
    similarity: float

@router.post("/query", response_model=List[QueryResponse])
async def query_vector_index(request: QueryRequest, db: Session = Depends(get_db)):
    inference_service = InferenceService()
    results = await inference_service.query_vector_index(request.query, request.node_ids)
    return [QueryResponse(**result) for result in results]