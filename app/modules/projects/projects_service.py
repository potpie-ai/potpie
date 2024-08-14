
from sqlalchemy.orm import Session

from app.core.base_service import BaseService
from app.modules.projects.projects_model import Project


class ProjectService(BaseService):
    def __init__(self, db: Session):
        super().__init__(db)

    def get_project_name(self, project_ids: list) -> str:
        projects = self.db.query(Project).filter(Project.id.in_(project_ids)).all()
        if not projects:
            raise ValueError("No valid projects found for the provided project IDs.")
        return projects[0].project_name if projects else "Unnamed Project"
