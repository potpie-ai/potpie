import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.projects.projects_service import ProjectService

logger = logging.getLogger(__name__)


class RunInferenceRequest(BaseModel):
    """Request body for run-inference endpoint."""
    use_inference_context: bool = True  # Use optimized context (85-90% token savings)
    force_rerun: bool = False  # Re-run even if docstrings already exist
    model_name: Optional[str] = None  # LLM model to use (default: from config)
    prompt_version: Optional[str] = None  # Prompt version for tracking


class ProjectController:
    @staticmethod
    async def get_project_list(
        user=Depends(AuthService.check_auth), db=Depends(get_db)
    ):
        user_id = user["user_id"]
        try:
            project_service = ProjectService(db)
            project_list = await project_service.list_projects(user_id)
            return project_list
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def delete_project(
        project_id: str, user=Depends(AuthService.check_auth), db=Depends(get_db)
    ):
        project_service = ProjectService(db)
        try:
            await project_service.delete_project(project_id)
            return JSONResponse(
                status_code=200,
                content={"message": "Project deleted successfully.", "id": project_id},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{str(e)}")

    @staticmethod
    async def run_inference(
        project_id: str,
        request: RunInferenceRequest,
        user=Depends(AuthService.check_auth),
        db=Depends(get_db)
    ):
        """
        Run inference (docstring generation) on an already-parsed project.

        This endpoint allows re-running inference without re-parsing:
        - After model upgrades
        - After prompt changes
        - To fill in missing docstrings
        - To improve docstring quality

        Returns immediately with session info; inference runs in background.
        """
        from app.modules.parsing.inference_session_model import InferenceSession
        from app.modules.parsing.inference_work_unit_model import InferenceWorkUnit
        from app.core.config_provider import config_provider
        from neo4j import GraphDatabase
        from celery import chord
        import os

        user_id = user["user_id"]
        project_service = ProjectService(db)

        try:
            # 1. Verify project exists and belongs to user
            project = project_service.get_project_from_db_by_id_sync(project_id)
            if not project:
                raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

            if project.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Not authorized to access this project")

            # 2. Check project status - must be parsed (READY or PARTIALLY_READY)
            status = project.get("status", "").upper()
            if status not in ("READY", "PARTIALLY_READY", "ERROR"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Project must be parsed first. Current status: {status}"
                )

            commit_id = project.get("commit_id", "unknown")

            # 3. Check for existing running inference session
            existing_session = db.query(InferenceSession).filter(
                InferenceSession.project_id == project_id,
                InferenceSession.status.in_(['pending', 'running'])
            ).first()

            if existing_session:
                return JSONResponse(
                    status_code=409,
                    content={
                        "message": "Inference already in progress",
                        "session_id": str(existing_session.id),
                        "status": existing_session.status,
                        "progress": f"{existing_session.completed_work_units}/{existing_session.total_work_units}"
                    }
                )

            # 4. Get next session number
            max_session = db.query(InferenceSession).filter(
                InferenceSession.project_id == project_id,
                InferenceSession.commit_id == commit_id
            ).order_by(InferenceSession.session_number.desc()).first()

            session_number = (max_session.session_number + 1) if max_session else 1

            # 5. Query Neo4j for directory structure and node counts
            neo4j_config = config_provider.get_neo4j_config()
            driver = GraphDatabase.driver(
                neo4j_config["uri"],
                auth=(neo4j_config["username"], neo4j_config["password"]),
            )

            try:
                with driver.session() as neo_session:
                    # Get total node count
                    total_result = neo_session.run("""
                        MATCH (n:NODE {repoId: $repo_id})
                        WHERE n.file_path IS NOT NULL AND n.file_path <> ''
                        RETURN count(n) AS total_nodes
                    """, repo_id=project_id).single()
                    total_nodes = total_result["total_nodes"] if total_result else 0

                    if total_nodes == 0:
                        raise HTTPException(
                            status_code=400,
                            detail="No parsed nodes found. Please parse the project first."
                        )

                    # Get directory counts
                    result = neo_session.run("""
                        MATCH (n:NODE {repoId: $repo_id})
                        WHERE n.file_path IS NOT NULL AND n.file_path <> ''
                        WITH n.file_path AS file_path
                        WITH
                            CASE
                                WHEN file_path CONTAINS '/'
                                THEN split(file_path, '/')[0..-1]
                                ELSE []
                            END AS path_parts
                        WITH
                            CASE
                                WHEN size(path_parts) = 0 THEN '<ROOT>'
                                ELSE reduce(p = '', part IN path_parts | p + '/' + part)
                            END AS directory
                        RETURN directory, count(*) AS node_count
                        ORDER BY node_count DESC
                    """, repo_id=project_id)

                    directory_counts = {
                        record["directory"]: record["node_count"]
                        for record in result
                    }
            finally:
                driver.close()

            # 6. Create InferenceSession
            inference_session = InferenceSession(
                id=uuid.uuid4(),
                project_id=project_id,
                commit_id=commit_id,
                session_number=session_number,
                total_work_units=0,  # Will update after creating work units
                total_nodes=total_nodes,
                use_inference_context=request.use_inference_context,
                model_name=request.model_name,
                prompt_version=request.prompt_version,
                status='pending',
                created_at=datetime.utcnow(),
            )
            db.add(inference_session)

            # 7. Build work units
            work_units = []
            max_nodes_per_unit = int(os.getenv('MAX_INFERENCE_NODES_PER_UNIT', '2000'))
            work_unit_index = 0

            for directory, node_count in directory_counts.items():
                is_root = (directory == '<ROOT>')
                directory_path = None if is_root else directory.lstrip('/')

                if node_count > max_nodes_per_unit:
                    # Split large directories
                    num_splits = (node_count // max_nodes_per_unit) + 1
                    for split_idx in range(num_splits):
                        work_unit = InferenceWorkUnit(
                            id=uuid.uuid4(),
                            project_id=project_id,
                            session_id=inference_session.id,
                            commit_id=commit_id,
                            work_unit_index=work_unit_index,
                            directory_path=directory_path or "/",
                            is_root=is_root,
                            node_count=node_count // num_splits,
                            split_index=split_idx,
                            total_splits=num_splits,
                            status='pending',
                            created_at=datetime.utcnow(),
                        )
                        db.add(work_unit)
                        work_units.append({
                            'project_id': project_id,
                            'user_id': user_id,
                            'directory_path': directory_path,
                            'is_root': is_root,
                            'split_index': split_idx,
                            'total_splits': num_splits,
                            'session_id': str(inference_session.id),
                            'work_unit_id': str(work_unit.id),
                            'use_inference_context': request.use_inference_context,
                            'filter_uninferred': not request.force_rerun,
                        })
                        work_unit_index += 1
                else:
                    work_unit = InferenceWorkUnit(
                        id=uuid.uuid4(),
                        project_id=project_id,
                        session_id=inference_session.id,
                        commit_id=commit_id,
                        work_unit_index=work_unit_index,
                        directory_path=directory_path or "/",
                        is_root=is_root,
                        node_count=node_count,
                        split_index=None,
                        total_splits=None,
                        status='pending',
                        created_at=datetime.utcnow(),
                    )
                    db.add(work_unit)
                    work_units.append({
                        'project_id': project_id,
                        'user_id': user_id,
                        'directory_path': directory_path,
                        'is_root': is_root,
                        'split_index': None,
                        'total_splits': None,
                        'session_id': str(inference_session.id),
                        'work_unit_id': str(work_unit.id),
                        'use_inference_context': request.use_inference_context,
                        'filter_uninferred': not request.force_rerun,
                    })
                    work_unit_index += 1

            # Update session with work unit count
            inference_session.total_work_units = len(work_units)
            db.commit()

            # 8. Spawn Celery chord
            from app.celery.tasks.parsing_tasks import (
                run_inference_unit,
                finalize_project_after_inference
            )

            queue_name = os.getenv('CELERY_QUEUE_NAME', 'staging') + '_process_repository'

            inference_chord = chord(
                [
                    run_inference_unit.signature(
                        kwargs=unit,
                        queue=queue_name
                    )
                    for unit in work_units
                ]
            )(
                finalize_project_after_inference.signature(
                    kwargs={
                        'project_id': project_id,
                        'user_id': user_id,
                        'session_id': str(inference_session.id),
                    },
                    queue=queue_name
                )
            )

            # Update session with coordinator task ID
            inference_session.coordinator_task_id = inference_chord.id
            inference_session.status = 'running'
            inference_session.started_at = datetime.utcnow()
            db.commit()

            logger.info(
                f"Started inference session {inference_session.id} for project {project_id}: "
                f"{len(work_units)} work units, chord_id={inference_chord.id}"
            )

            return JSONResponse(
                status_code=202,
                content={
                    "message": "Inference started",
                    "session_id": str(inference_session.id),
                    "project_id": project_id,
                    "total_work_units": len(work_units),
                    "total_nodes": total_nodes,
                    "use_inference_context": request.use_inference_context,
                    "chord_id": inference_chord.id,
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error starting inference for project {project_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def get_inference_status(
        project_id: str,
        user=Depends(AuthService.check_auth),
        db=Depends(get_db)
    ):
        """
        Get inference status for a project.

        Returns the most recent inference session and its work units.
        """
        from app.modules.parsing.inference_session_model import InferenceSession
        from app.modules.parsing.inference_work_unit_model import InferenceWorkUnit

        user_id = user["user_id"]
        project_service = ProjectService(db)

        try:
            # Verify project exists and belongs to user
            project = project_service.get_project_from_db_by_id_sync(project_id)
            if not project:
                raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

            if project.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Not authorized to access this project")

            # Get the most recent inference session
            latest_session = db.query(InferenceSession).filter(
                InferenceSession.project_id == project_id
            ).order_by(InferenceSession.created_at.desc()).first()

            if not latest_session:
                return JSONResponse(
                    status_code=200,
                    content={
                        "has_inference": False,
                        "message": "No inference sessions found for this project"
                    }
                )

            # Get work unit summary
            work_units = db.query(InferenceWorkUnit).filter(
                InferenceWorkUnit.session_id == latest_session.id
            ).all()

            work_unit_summary = {
                'pending': sum(1 for w in work_units if w.status == 'pending'),
                'processing': sum(1 for w in work_units if w.status == 'processing'),
                'completed': sum(1 for w in work_units if w.status == 'completed'),
                'failed': sum(1 for w in work_units if w.status == 'failed'),
                'skipped': sum(1 for w in work_units if w.status == 'skipped'),
            }

            # Build failed work unit details
            failed_details = [
                {
                    'work_unit_id': str(w.id),
                    'directory_path': w.directory_path,
                    'error_message': w.error_message,
                    'error_type': w.error_type,
                    'attempt_count': w.attempt_count,
                }
                for w in work_units if w.status == 'failed'
            ]

            return JSONResponse(
                status_code=200,
                content={
                    "has_inference": True,
                    "session": {
                        "session_id": str(latest_session.id),
                        "session_number": latest_session.session_number,
                        "status": latest_session.status,
                        "total_work_units": latest_session.total_work_units,
                        "completed_work_units": latest_session.completed_work_units,
                        "failed_work_units": latest_session.failed_work_units,
                        "total_nodes": latest_session.total_nodes,
                        "processed_nodes": latest_session.processed_nodes,
                        "use_inference_context": latest_session.use_inference_context,
                        "model_name": latest_session.model_name,
                        "created_at": latest_session.created_at.isoformat() if latest_session.created_at else None,
                        "started_at": latest_session.started_at.isoformat() if latest_session.started_at else None,
                        "completed_at": latest_session.completed_at.isoformat() if latest_session.completed_at else None,
                        "completion_percentage": latest_session.completion_percentage(),
                        "is_resumable": latest_session.is_resumable(),
                        "error_message": latest_session.error_message,
                    },
                    "work_units": work_unit_summary,
                    "failed_work_units": failed_details[:10],  # Limit to 10
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error getting inference status for project {project_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def resume_inference(
        project_id: str,
        user=Depends(AuthService.check_auth),
        db=Depends(get_db)
    ):
        """
        Resume inference by retrying failed work units.

        Only works if there's a resumable session (status: failed, partial, or paused).
        Creates new tasks only for failed/pending work units.
        """
        from app.modules.parsing.inference_session_model import InferenceSession
        from app.modules.parsing.inference_work_unit_model import InferenceWorkUnit
        from celery import group
        import os

        user_id = user["user_id"]
        project_service = ProjectService(db)

        try:
            # Verify project
            project = project_service.get_project_from_db_by_id_sync(project_id)
            if not project:
                raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

            if project.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Not authorized")

            # Get most recent session
            session = db.query(InferenceSession).filter(
                InferenceSession.project_id == project_id
            ).order_by(InferenceSession.created_at.desc()).first()

            if not session:
                raise HTTPException(status_code=404, detail="No inference session found")

            if not session.is_resumable():
                raise HTTPException(
                    status_code=400,
                    detail=f"Session is not resumable (status: {session.status})"
                )

            # Get failed/pending work units
            retriable_units = db.query(InferenceWorkUnit).filter(
                InferenceWorkUnit.session_id == session.id,
                InferenceWorkUnit.status.in_(['failed', 'pending'])
            ).all()

            if not retriable_units:
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "No work units to retry",
                        "session_id": str(session.id)
                    }
                )

            # Check retry limits
            units_to_retry = [u for u in retriable_units if u.is_retriable()]
            if not units_to_retry:
                raise HTTPException(
                    status_code=400,
                    detail="All failed units have exceeded max retry attempts"
                )

            # Build task signatures
            from app.celery.tasks.parsing_tasks import run_inference_unit

            queue_name = os.getenv('CELERY_QUEUE_NAME', 'staging') + '_process_repository'

            tasks = []
            for unit in units_to_retry:
                tasks.append(
                    run_inference_unit.signature(
                        kwargs={
                            'project_id': project_id,
                            'user_id': user_id,
                            'directory_path': unit.directory_path if unit.directory_path != "/" else None,
                            'is_root': unit.is_root,
                            'split_index': unit.split_index,
                            'total_splits': unit.total_splits,
                            'session_id': str(session.id),
                            'work_unit_id': str(unit.id),
                            'use_inference_context': session.use_inference_context,
                            'filter_uninferred': True,  # Only process missing docstrings
                        },
                        queue=queue_name
                    )
                )

            # Reset work unit statuses
            for unit in units_to_retry:
                unit.status = 'pending'
                unit.error_message = None
                unit.error_type = None
            db.commit()

            # Update session status
            session.status = 'running'
            session.started_at = datetime.utcnow()
            db.commit()

            # Dispatch tasks as a group (no callback - session already exists)
            task_group = group(tasks)
            result = task_group.apply_async()

            logger.info(
                f"Resumed inference session {session.id}: retrying {len(units_to_retry)} work units"
            )

            return JSONResponse(
                status_code=202,
                content={
                    "message": "Inference resumed",
                    "session_id": str(session.id),
                    "work_units_retrying": len(units_to_retry),
                    "group_id": result.id,
                }
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error resuming inference for project {project_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
