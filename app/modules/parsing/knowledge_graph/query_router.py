from fastapi import APIRouter, Depends
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.parsing.knowledge_graph.inference_schema import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/query", response_model=List[QueryResponse])
async def query_vector_index(request: QueryRequest, db: Session = Depends(get_db)):
    inference_service = InferenceService()
    results = await inference_service.query_vector_index(request.project_id, request.query, request.node_ids)
    return [QueryResponse(**result) for result in results]