import os
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


from fastapi import Header, HTTPException


@router.post("/query", response_model=List[QueryResponse])
async def query_vector_index(
    request: QueryRequest,
    db: Session = Depends(get_db),
    authorization: str = Header(None),
):
    INTERNAL_CALL_SECRET = os.getenv("INTERNAL_CALL_SECRET")

    if authorization != f"Bearer {INTERNAL_CALL_SECRET}":
        raise HTTPException(status_code=403, detail="Invalid authorization token")

    inference_service = InferenceService()
    results = await inference_service.query_vector_index(
        request.project_id, request.query, request.node_ids
    )
    return [QueryResponse(**result) for result in results]
