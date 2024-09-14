import logging
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import String, cast
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.modules.projects.projects_model import Project
from app.modules.projects.projects_schema import ProjectStatusEnum

logger = logging.getLogger(__name__)


class ProjectServiceError(Exception):
    """Base exception class for ProjectService errors."""


class ProjectNotFoundError(ProjectServiceError):
    """Raised when a project is not found."""


class ProjectService:
    def __init__(self, db: Session):
        self.db = db

    async def get_project_name(self, project_ids: list) -> str:
        try:
            projects = self.db.query(Project).filter(Project.id.in_(project_ids)).all()
            if not projects:
                raise ProjectNotFoundError(
                    "No valid projects found for the provided project IDs."
                )
            project_name = projects[0].repo_name
            logger.info(
                f"Retrieved project name: {project_name} for project IDs: {project_ids}"
            )
            return project_name
        except SQLAlchemyError as e:
            logger.error(
                f"Database error in get_project_name for project IDs {project_ids}: {e}",
                exc_info=True,
            )
            raise ProjectServiceError(
                f"Failed to retrieve project name for project IDs {project_ids}"
            ) from e
        except ProjectNotFoundError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in get_project_name for project IDs {project_ids}: {e}",
                exc_info=True,
            )
            raise ProjectServiceError(
                f"An unexpected error occurred while retrieving project name for project IDs {project_ids}"
            ) from e

    async def register_project(
        self, repo_name: str, branch_name: str, user_id: str, project_id: str
    ):
        project = Project(
            id=project_id,
            repo_name=repo_name,
            branch_name=branch_name,
            user_id=user_id,
            status=ProjectStatusEnum.SUBMITTED.value,
        )
        project = ProjectService.create_project(self.db, project)
        message = f"Project id '{project.id}' for repo '{repo_name}' and branch '{branch_name}' registered successfully."
        logging.info(message)
        return project_id

    async def list_projects(self, user_id: str):
        projects = ProjectService.get_projects_by_user_id(self.db, user_id)
        project_list = []
        for project in projects:
            project_dict = {
                "id": project.id,
                "repo_name": project.repo_name,
                "status": project.status,
            }
            project_list.append(project_dict)
        return project_list

    async def update_project_status(self, project_id: int, status: ProjectStatusEnum):
        ProjectService.update_project(self.db, project_id, status=status.value)
        logging.info(
            f"Project with ID {project_id} has now been updated with status {status}."
        )

    async def get_project_from_db(self, repo_name: str, user_id: str):
        project = (
            self.db.query(Project)
            .filter(Project.repo_name == repo_name, Project.user_id == user_id)
            .first()
        )
        if project:
            return project
        else:
            return None

    async def get_project_from_db_by_id(self, project_id: int):
        project = ProjectService.get_project_by_id(self.db, project_id)
        if project:
            return {
                "project_name": project.repo_name,
                "id": project.id,
                "commit_id": project.commit_id,
                "status": project.status,
                "branch_name": project.branch_name,
            }
        else:
            return None

    async def get_project_repo_details_from_db(self, project_id: int, user_id: str):
        project = (
            self.db.query(Project)
            .filter(Project.id == project_id, Project.user_id == user_id)
            .first()
        )
        if project:
            return {
                "project_name": project.project_name,
                "directory": project.directory,
                "id": project.id,
                "repo_name": project.repo_name,
                "branch_name": project.branch_name,
            }
        else:
            return None

    async def get_repo_and_branch_name(self, project_id: int):
        project = ProjectService.get_project_by_id(self.db, project_id)
        if project:
            return project.repo_name, project.branch_name, project.directory
        else:
            return None

    async def get_project_from_db_by_id_and_user_id(
        self, project_id: int, user_id: str
    ):
        project = (
            self.db.query(Project)
            .filter(Project.id == project_id, Project.user_id == user_id)
            .first()
        )
        if project:
            return {
                "id": project.id,
                "commit_id": project.commit_id,
                "status": project.status,
            }
        else:
            return None

    def get_project_by_id(db: Session, project_id: int):
        return db.query(Project).filter(Project.id == project_id).first()

    def get_projects_by_user_id(db: Session, user_id: str):
        return db.query(Project).filter(Project.user_id == user_id).all()

    def create_project(db: Session, project: Project):
        project.created_at = datetime.utcnow()
        project.updated_at = datetime.utcnow()
        db.add(project)
        db.commit()
        db.refresh(project)
        return project

    def update_project(db: Session, project_id: int, **kwargs):
        project = db.query(Project).filter(Project.id == project_id).first()

        if project is None:
            return None  # Project doesn't exist

        result = db.query(Project).filter(Project.id == project_id).update(kwargs)

        if result > 0:
            db.commit()
            return result

        return None

    def delete_project(self, project_id: int):
        project = (
            self.db.query(Project)
            .filter(cast(Project.id, String) == str(project_id))
            .first()
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")
        self.db.delete(project)
        self.db.commit()
