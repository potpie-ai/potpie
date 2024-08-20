import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.modules.projects.projects_model import Project

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
                raise ProjectNotFoundError("No valid projects found for the provided project IDs.")
            project_name = projects[0].project_name if projects else "Unnamed Project"
            logger.info(f"Retrieved project name: {project_name} for project IDs: {project_ids}")
            return project_name
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_project_name for project IDs {project_ids}: {e}", exc_info=True)
            raise ProjectServiceError(f"Failed to retrieve project name for project IDs {project_ids}") from e
        except ProjectNotFoundError as e:
            logger.warning(str(e))
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_project_name for project IDs {project_ids}: {e}", exc_info=True)
            raise ProjectServiceError(f"An unexpected error occurred while retrieving project name for project IDs {project_ids}") from e