from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService

from .search_schema import SearchRequest, SearchResponse
from .search_service import SearchService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_codebase(
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    search_service = SearchService(db)
    results = await search_service.search_codebase(
        search_request.project_id, search_request.query
    )
    return SearchResponse(results=results)
