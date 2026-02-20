import asyncio
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import String, cast
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProjectServiceError(Exception):
    """Base exception class for ProjectService errors."""


class ProjectNotFoundError(ProjectServiceError):
    """Raised when a project is not found."""


class ProjectService:
    def __init__(self, db: Session):
        self.db = db

    @classmethod
    def create(cls, db: Session) -> "ProjectService":
        """Factory method for creating a ProjectService instance."""
        return cls(db)

    @classmethod
    def create_from_config(
        cls,
        db: Session,
        *,
        raise_library_exceptions: bool = False,
    ) -> "ProjectService":
        """Factory method that accepts explicit config for library usage.

        This method creates a ProjectService configured for library usage,
        optionally raising library-specific exceptions instead of HTTPException.

        Args:
            db: Database session
            raise_library_exceptions: If True, raise ProjectServiceError/ProjectNotFoundError
                                      instead of HTTPException (for library usage)

        Returns:
            Configured ProjectService instance
        """
        instance = cls(db)
        instance._raise_library_exceptions = raise_library_exceptions
        return instance

    async def get_project_name(self, project_ids: list) -> str:
        def _run():
            db = SessionLocal()
            try:
                projects = db.query(Project).filter(Project.id.in_(project_ids)).all()
                if not projects:
                    raise ProjectNotFoundError(
                        "No valid projects found for the provided project IDs."
                    )
                return projects[0].repo_name
            finally:
                db.close()

        try:
            project_name = await asyncio.to_thread(_run)
            logger.info(
                f"Retrieved project name: {project_name} for project IDs: {project_ids}"
            )
            return project_name
        except ProjectNotFoundError as e:
            logger.warning(str(e))
            raise
        except SQLAlchemyError as e:
            logger.error(
                f"Database error in get_project_name for project IDs {project_ids}: {e}",
                exc_info=True,
            )
            raise ProjectServiceError(
                f"Failed to retrieve project name for project IDs {project_ids}"
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error in get_project_name for project IDs {project_ids}: {e}",
                exc_info=True,
            )
            raise ProjectServiceError(
                f"An unexpected error occurred while retrieving project name for project IDs {project_ids}"
            ) from e

    async def register_project(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        project_id: str,
        commit_id: str | None = None,
        repo_path: str | None = None,
    ):
        def _run():
            db = SessionLocal()
            try:
                existing_project = (
                    db.query(Project).filter(Project.id == project_id).first()
                )
                if existing_project:
                    if existing_project.user_id != user_id:
                        message = (
                            f"Project {project_id} ownership mismatch: "
                            f"stored user {existing_project.user_id}, requesting user {user_id}"
                        )
                        logger.warning(message)
                        if getattr(self, "_raise_library_exceptions", False):
                            raise ProjectServiceError(message)
                        raise HTTPException(status_code=403, detail=message)
                    logger.info(
                        f"Project {project_id} already exists. Updating repo_name from '{existing_project.repo_name}' to '{repo_name}'"
                    )
                    existing_project.repo_name = repo_name
                    existing_project.branch_name = branch_name
                    existing_project.repo_path = repo_path
                    existing_project.commit_id = commit_id
                    existing_project.status = ProjectStatusEnum.SUBMITTED.value
                    existing_project.updated_at = datetime.utcnow()
                    try:
                        db.commit()
                        db.refresh(existing_project)
                    except Exception:
                        logger.exception(f"Error updating existing project {project_id}")
                        db.rollback()
                        raise
                    logger.info(
                        f"Project id '{project_id}' for repo '{repo_name}' and branch '{branch_name}' updated successfully."
                    )
                    return project_id
                project = Project(
                    id=project_id,
                    repo_name=repo_name,
                    branch_name=branch_name,
                    user_id=user_id,
                    repo_path=repo_path,
                    commit_id=commit_id,
                    status=ProjectStatusEnum.SUBMITTED.value,
                )
                try:
                    project = ProjectService.create_project(db, project)
                except Exception as e:
                    logger.error(f"Error creating project {project_id}: {e}")
                    db.rollback()
                    raise
                logger.info(
                    f"Project id '{project.id}' for repo '{repo_name}' and branch '{branch_name}' registered successfully."
                )
                return project_id
            finally:
                db.close()

        return await asyncio.to_thread(_run)

    async def duplicate_project(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        project_id: str,
        properties,
        commit_id,
    ):
        def _run():
            db = SessionLocal()
            try:
                project = Project(
                    id=project_id,
                    repo_name=repo_name,
                    branch_name=branch_name,
                    user_id=user_id,
                    properties=properties,
                    commit_id=commit_id,
                    status=ProjectStatusEnum.SUBMITTED.value,
                )
                ProjectService.create_project(db, project)
                return project_id
            finally:
                db.close()

        result = await asyncio.to_thread(_run)
        logger.info(
            f"Project id '{project_id}' for repo '{repo_name}' and branch '{branch_name}' registered successfully."
        )
        return result

    async def list_projects(self, user_id: str):
        def _run():
            db = SessionLocal()
            try:
                projects = ProjectService.get_projects_by_user_id(db, user_id)
                return [
                    {"id": p.id, "repo_name": p.repo_name, "status": p.status}
                    for p in projects
                ]
            finally:
                db.close()

        return await asyncio.to_thread(_run)

    async def update_project_status(self, project_id: str, status: ProjectStatusEnum):
        def _run():
            db = SessionLocal()
            try:
                ProjectService.update_project(db, project_id, status=status.value)
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        try:
            await asyncio.to_thread(_run)
            logger.info(
                f"Project with ID {project_id} has now been updated with status {status}."
            )
        except Exception:
            logger.exception(f"Error updating project status for {project_id}")
            raise

    @staticmethod
    def _query_project_from_db(
        db: Session,
        repo_name: str,
        branch_name: str | None,
        user_id: str,
        repo_path: str | None = None,
        commit_id: str | None = None,
    ):
        """Sync query for get_project_from_db. Use from thread with a dedicated session."""
        query = db.query(Project).filter(
            Project.repo_name == repo_name,
            Project.user_id == user_id,
            Project.repo_path == repo_path,
        )
        if commit_id:
            return query.filter(Project.commit_id == commit_id).first()
        return query.filter(Project.branch_name == branch_name).first()

    async def get_project_from_db(
        self,
        repo_name: str,
        branch_name: str | None,
        user_id: str,
        repo_path: str | None = None,
        commit_id: str | None = None,
    ):
        """
        Get a project from the database for a specific user, prioritizing commit_id over branch_name.
        """
        logger.info(
            f"Looking up project: repo_name={repo_name}, branch={branch_name}, "
            f"user={user_id}, repo_path={repo_path}, commit_id={commit_id}"
        )

        def _run():
            db = SessionLocal()
            try:
                return ProjectService._query_project_from_db(
                    db, repo_name, branch_name, user_id, repo_path, commit_id
                )
            finally:
                db.close()

        project = await asyncio.to_thread(_run)
        if project:
            logger.info(
                f"Found project by {'commit_id' if commit_id else 'branch_name'}: {project.id}"
            )
        else:
            logger.info(
                "No existing project found for this repository and branch"
                + (f" (commit_id={commit_id})" if commit_id else "")
            )
        return project

    async def get_global_project_from_db(
        self,
        repo_name: str,
        branch_name: str,
        repo_path: str | None = None,
        commit_id: str | None = None,
    ):
        """Get a global project from the database, prioritizing commit_id over branch_name."""

        def _run():
            db = SessionLocal()
            try:
                query = db.query(Project).filter(
                    Project.repo_name == repo_name,
                    Project.status == ProjectStatusEnum.READY.value,
                    Project.repo_path == repo_path,
                )
                if commit_id:
                    project = (
                        query.filter(Project.commit_id == commit_id)
                        .order_by(Project.created_at.asc())
                        .first()
                    )
                    if project:
                        return project
                return (
                    query.filter(Project.branch_name == branch_name)
                    .order_by(Project.created_at.asc())
                    .first()
                )
            finally:
                db.close()

        return await asyncio.to_thread(_run)

    async def get_project_from_db_by_id(self, project_id: str):
        def _run():
            db = SessionLocal()
            try:
                project = ProjectService.get_project_by_id(db, project_id)
                if project:
                    return {
                        "project_name": project.repo_name,
                        "id": project.id,
                        "commit_id": project.commit_id,
                        "status": project.status,
                        "branch_name": project.branch_name,
                        "user_id": project.user_id,
                        "repo_path": project.repo_path,
                    }
                return None
            finally:
                db.close()

        return await asyncio.to_thread(_run)

    def get_project_from_db_by_id_sync(self, project_id: int):
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if project:
            return {
                "project_name": project.repo_name,
                "id": project.id,
                "commit_id": project.commit_id,
                "status": project.status,
                "branch_name": project.branch_name,
                "repo_path": project.repo_path,
                "user_id": project.user_id,
            }
        else:
            return None

    async def get_project_repo_details_from_db(self, project_id: int, user_id: str):
        def _run():
            db = SessionLocal()
            try:
                project = (
                    db.query(Project)
                    .filter(Project.id == project_id, Project.user_id == user_id)
                    .first()
                )
                if project:
                    return {
                        "id": project.id,
                        "repo_name": project.repo_name,
                        "branch_name": project.branch_name,
                        "user_id": project.user_id,
                        "repo_path": project.repo_path,
                        "commit_id": project.commit_id,
                    }
                return None
            finally:
                db.close()

        return await asyncio.to_thread(_run)

    async def get_repo_and_branch_name(self, project_id: int):
        def _run():
            db = SessionLocal()
            try:
                project = ProjectService.get_project_by_id(db, project_id)
                if project:
                    return (
                        project.repo_name,
                        project.branch_name,
                        project.directory,
                        project.repo_path,
                    )
                return None
            finally:
                db.close()

        return await asyncio.to_thread(_run)

    async def get_project_from_db_by_id_and_user_id(
        self, project_id: int, user_id: str
    ):
        def _run():
            db = SessionLocal()
            try:
                project = (
                    db.query(Project)
                    .filter(Project.id == project_id, Project.user_id == user_id)
                    .first()
                )
                if project:
                    return {
                        "id": project.id,
                        "commit_id": project.commit_id,
                        "status": project.status,
                    }
                return None
            finally:
                db.close()

        return await asyncio.to_thread(_run)

    def get_project_by_id(db: Session, project_id: int):
        return db.query(Project).filter(Project.id == project_id).first()

    def get_projects_by_user_id(db: Session, user_id: str):
        return db.query(Project).filter(Project.user_id == user_id).all()

    def create_project(db: Session, project: Project):
        project.created_at = datetime.utcnow()
        project.updated_at = datetime.utcnow()
        db.add(project)
        try:
            db.commit()
            db.refresh(project)
            return project
        except IntegrityError:
            db.rollback()
            logger.exception(f"IntegrityError creating project {project.id}")
            raise
        except Exception:
            db.rollback()
            logger.exception(f"Error creating project {project.id}")
            raise

    @staticmethod
    def update_project(db, project_id: str, **kwargs):
        project = db.query(Project).filter(Project.id == project_id).first()

        if project is None:
            return None  # Project doesn't exist

        result = db.query(Project).filter(Project.id == project_id).update(kwargs)

        if result > 0:
            db.commit()
            return result

        return None

    async def delete_project(self, project_id: str):
        raise_lib = getattr(self, "_raise_library_exceptions", False)

        def _run():
            db = SessionLocal()
            try:
                project = (
                    db.query(Project)
                    .filter(cast(Project.id, String) == str(project_id))
                    .first()
                )
                if not project:
                    if raise_lib:
                        raise ProjectNotFoundError(f"Project not found: {project_id}")
                    raise HTTPException(status_code=404, detail="Project not found.")
                db.delete(project)
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        await asyncio.to_thread(_run)

    async def get_demo_project_id(self, repo_name: str):
        def _run():
            db = SessionLocal()
            try:
                project = (
                    db.query(Project).filter(Project.repo_name == repo_name).first()
                )
                return project.id if project else None
            finally:
                db.close()

        try:
            project_id = await asyncio.to_thread(_run)
            if project_id:
                logger.info(
                    f"Retrieved demo repo ID: {project_id} for repo name: {repo_name}"
                )
                return project_id
            raise ProjectNotFoundError(
                f"No demo repository found for repo name: {repo_name}"
            )
        except (ProjectNotFoundError, ProjectServiceError):
            raise
        except SQLAlchemyError as e:
            logger.error(
                f"Database error in get_demo_repo_id for repo name {repo_name}: {e}",
                exc_info=True,
            )
            raise ProjectServiceError(
                f"Failed to retrieve demo repo ID for repo name {repo_name}"
            ) from e
        except Exception as e:
            logger.error(
                f"Unexpected error in get_demo_repo_id for repo name {repo_name}: {e}",
                exc_info=True,
            )
            raise ProjectServiceError(
                f"An unexpected error occurred while retrieving demo repo ID for repo name {repo_name}"
            ) from e
