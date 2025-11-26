from fastapi import Depends

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.utils.APIRouter import APIRouter

from .projects_controller import ProjectController, RunInferenceRequest

router = APIRouter()


@router.get("/projects/list")
async def get_project_list(user=Depends(AuthService.check_auth), db=Depends(get_db)):
    return await ProjectController.get_project_list(user=user, db=db)


@router.delete("/projects")
async def delete_project(
    project_id: str, user=Depends(AuthService.check_auth), db=Depends(get_db)
):
    return await ProjectController.delete_project(
        project_id=project_id, user=user, db=db
    )


@router.post("/projects/{project_id}/run-inference")
async def run_inference(
    project_id: str,
    request: RunInferenceRequest = RunInferenceRequest(),
    user=Depends(AuthService.check_auth),
    db=Depends(get_db)
):
    """
    Run inference (docstring generation) on an already-parsed project.

    This endpoint allows re-running inference without re-parsing:
    - After model upgrades (change model_name)
    - After prompt changes (track with prompt_version)
    - To fill in missing docstrings (default: filter_uninferred=True)
    - To regenerate all docstrings (force_rerun=True)

    Request body:
    - use_inference_context: Use optimized context for 85-90% token savings (default: True)
    - force_rerun: Regenerate all docstrings, not just missing ones (default: False)
    - model_name: LLM model to use (optional, uses config default)
    - prompt_version: Version string for tracking prompt changes (optional)

    Returns 202 Accepted with session info. Check status via GET /projects/{project_id}/inference-status.
    """
    return await ProjectController.run_inference(
        project_id=project_id,
        request=request,
        user=user,
        db=db
    )


@router.get("/projects/{project_id}/inference-status")
async def get_inference_status(
    project_id: str,
    user=Depends(AuthService.check_auth),
    db=Depends(get_db)
):
    """
    Get inference status for a project.

    Returns the most recent inference session status including:
    - Session progress (completed/total work units)
    - Work unit breakdown by status
    - Failed work unit details (for debugging)
    - Whether the session is resumable

    Use this to monitor inference progress or debug failures.
    """
    return await ProjectController.get_inference_status(
        project_id=project_id,
        user=user,
        db=db
    )


@router.post("/projects/{project_id}/resume-inference")
async def resume_inference(
    project_id: str,
    user=Depends(AuthService.check_auth),
    db=Depends(get_db)
):
    """
    Resume inference by retrying failed work units.

    Prerequisites:
    - A previous inference session must exist
    - Session must be resumable (status: failed, partial, or paused)
    - At least one work unit must be retriable (under max_attempts)

    This endpoint:
    1. Finds failed/pending work units from the most recent session
    2. Resets their status to pending
    3. Dispatches new Celery tasks for each
    4. Returns immediately (tasks run in background)

    Use GET /inference-status to monitor progress.
    """
    return await ProjectController.resume_inference(
        project_id=project_id,
        user=user,
        db=db
    )
