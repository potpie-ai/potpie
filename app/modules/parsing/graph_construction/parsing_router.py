from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.parsing.graph_construction.parsing_controller import ParsingController
from app.modules.parsing.graph_construction.parsing_schema import (
    ParsingRequest,
    ParsingStatusRequest,
)
from app.modules.utils.APIRouter import APIRouter

router = APIRouter()


@router.post("/parse")
async def parse_directory(
    repo_details: ParsingRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    return await ParsingController.parse_directory(repo_details, db, user)


@router.get("/parsing-status/{project_id}")
async def get_parsing_status(
    project_id: str, db: Session = Depends(get_db), user=Depends(AuthService.check_auth)
):
    return await ParsingController.fetch_parsing_status(project_id, db, user)


@router.post("/parsing-status")
async def get_parsing_status_by_repo(
    request: ParsingStatusRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    return await ParsingController.fetch_parsing_status_by_repo(request, db, user)


@router.post("/check-status")
async def check_parsing_status(
    request: ParsingStatusRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    """Check if a repository/branch is already parsed (alias for parsing-status)"""
    return await ParsingController.fetch_parsing_status_by_repo(request, db, user)
