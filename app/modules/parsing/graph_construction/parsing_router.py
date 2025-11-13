import logging
import time

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.parsing.graph_construction.parsing_controller import ParsingController
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.utils.APIRouter import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/parse")
async def parse_directory(
    repo_details: ParsingRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    start_time = time.perf_counter()
    logger.info(
        f"[TIMING] parsing_router.parse_directory: START | "
        f"repo_name={repo_details.repo_name}, branch={repo_details.branch_name}, "
        f"commit_id={repo_details.commit_id}"
    )
    try:
        result = await ParsingController.parse_directory(repo_details, db, user)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[TIMING] parsing_router.parse_directory: END | "
            f"elapsed={elapsed:.4f}s | repo_name={repo_details.repo_name}"
        )
        return result
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(
            f"[TIMING] parsing_router.parse_directory: ERROR | "
            f"elapsed={elapsed:.4f}s | error={str(e)}"
        )
        raise


@router.get("/parsing-status/{project_id}")
async def get_parsing_status(
    project_id: str, db: Session = Depends(get_db), user=Depends(AuthService.check_auth)
):
    start_time = time.perf_counter()
    logger.info(f"[TIMING] parsing_router.get_parsing_status: START | project_id={project_id}")
    try:
        result = await ParsingController.fetch_parsing_status(project_id, db, user)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[TIMING] parsing_router.get_parsing_status: END | "
            f"elapsed={elapsed:.4f}s | project_id={project_id}"
        )
        return result
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(
            f"[TIMING] parsing_router.get_parsing_status: ERROR | "
            f"elapsed={elapsed:.4f}s | error={str(e)}"
        )
        raise
