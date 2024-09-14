from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.parsing.knowledge_graph.inference_schema import (
    QueryRequest,
    QueryResponse,
)
from app.modules.parsing.knowledge_graph.inference_service import InferenceService

router = APIRouter()


@router.post("/query", response_model=List[QueryResponse])
async def query_vector_index(request: QueryRequest, db: Session = Depends(get_db)):
    inference_service = InferenceService(db)
    results = await inference_service.query_vector_index(
        request.project_id, request.query, request.node_ids
    )
    return [QueryResponse(**result) for result in results]
