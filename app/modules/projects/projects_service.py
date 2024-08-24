import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.modules.projects.projects_model import Project
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.utils.model_helper import ModelHelper
import logging
from fastapi import HTTPException
from datetime import datetime
import os
import shutil
from sqlalchemy import and_

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

    
    async def register_project(self, repo_name: str, branch_name: str, user_id: str, project_id: str):
        
        
        
        project = Project(id=project_id, repo_name=repo_name,
                                branch_name=branch_name, user_id=user_id,
                                 status=ProjectStatusEnum.SUBMITTED.value)
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
                "directory": project.directory,
                "active": project.is_default,
            }
            project_list.append(project_dict)
        return project_list

    async def update_project_status(self, project_id: int, status: ProjectStatusEnum):
        ProjectService.update_project(self.db, project_id, status=status.value)
        logging.info(f"Project with ID {project_id} has now been updated with status {status}.")


    async def get_project_from_db(self, repo_name: str, user_id: str):
        project = self.db.query(Project).filter(Project.repo_name == repo_name, Project.user_id == user_id).first()
        if project:
            return project
        else:
            return None

    async def get_project_from_db_by_id(self, project_id: int):
        project = ProjectService.get_project_by_id(self.db, project_id)
        if project:
            return {
                "project_name": project.project_name,
                "directory": project.directory,
                "id": project.id,
                "commit_id": project.commit_id,
                "status": project.status
            }
        else:
            return None

    async def get_project_repo_details_from_db(self, project_id: int, user_id: str):
        project = self.db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
        if project:
            return {
                "project_name": project.project_name,
                "directory": project.directory,
                "id": project.id,
                "repo_name": project.repo_name,
                "branch_name": project.branch_name
            }
        else:
            return None

    async def get_repo_and_branch_name(self, project_id: int):
        project = ProjectService.get_project_by_id(self.db, project_id)
        if project:
            return project.repo_name, project.branch_name , project.directory
        else:
            return None

    async def get_project_from_db_by_id_and_user_id(self, project_id: int, user_id: str):
        project = self.db.query(Project).filter(Project.id == project_id, Project.user_id == user_id).first()
        if project:
            return {
                'project_name': project.project_name,
                'directory': project.directory,
                'id': project.id,
                'commit_id': project.commit_id,
                'status': project.status
            }
        else:
            return None

    async def get_first_project_from_db_by_repo_name_branch_name(self, repo_name, branch_name):
        project = self.db.query(Project).filter(Project.repo_name == repo_name, Project.branch_name == branch_name).first()
        if project:
            return ModelHelper.model_to_dict(project)
        else:
            return None

    async def get_first_user_id_from_project_repo_name(self, repo_name):
        project = self.db.query(Project).filter(Project.repo_name == repo_name).first()
        if project:
            return project.user_id
        else:
            return None

    async def get_parsed_project_branches(self, repo_name: str = None, user_id: str = None, default: bool = None):
        query = self.db.query(Project).filter(Project.user_id == user_id)
        if default is not None:
            query = query.filter(Project.is_default == default)
        if repo_name is not None:
            query = query.filter(Project.repo_name == repo_name)
        projects = query.all()
        return [(p.id, p.branch_name, p.repo_name, p.updated_at, p.is_default, p.status) for p in projects]


    async def delete_project(self, project_id: int, user_id: str):
        try:
            result = ProjectService.update_project(
                self.db,
                project_id,
                is_deleted=True,
                updated_at=datetime.utcnow(),
                user_id=user_id
            )
            if not result:
                raise HTTPException(
                    status_code=404,
                    detail="No matching project found or project is already deleted."
                )
            else:
                is_local_repo = os.getenv("isDevelopmentMode") == "enabled" and user_id == os.getenv("defaultUsername")
                if is_local_repo:
                    project_path = self.get_project_repo_details_from_db(project_id,user_id)['directory']
                    if os.path.exists(project_path):
                        shutil.rmtree(project_path)
                        logging.info(f"Deleted local project folder: {project_path}")
                    else:
                        logging.warning(f"Local project folder not found: {project_path}")


                logging.info(f"Project {project_id} deleted successfully.")

        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while deleting the project: {str(e)}"
            )

    async def restore_project(self, project_id: int, user_id: str):
        try:
            result = ProjectService.update_project(
                self.db,
                project_id,
                is_deleted=False,
                user_id=user_id
            )
            if result:
                message = f"Project with ID {project_id} restored successfully."
            else:
                message = "Project not found or already restored."
            logging.info(message)
            return message
        except Exception as e:
            self.db.rollback()
            logging.error(f"An error occurred: {e}")
            return "Error occurred during restoration."

    async def restore_all_project(self, repo_name: str, user_id: str):
        try:
            projects = ProjectService.get_projects_by_repo_name(self.db, repo_name, user_id, is_deleted=True)
            for project in projects:
                ProjectService.update_project(self.db, project.id, is_deleted=False)
            if projects:
                message = f"Projects with repo_name {repo_name} restored successfully."
            else:
                message = "Projects not found or already restored."
            logging.info(message)
            return message
        except Exception as e:
            self.db.rollback()
            logging.error(f"An error occurred: {e}")
            return "Error occurred during restoration."

    async def delete_all_project_by_repo_name(self, repo_name: str, user_id: str):
        try:
            projects = ProjectService.get_projects_by_repo_name(self.db, repo_name, user_id, is_deleted=False)
            for project in projects:
                ProjectService.update_project(self.db, project.id, is_deleted=True)
            if projects:
                message = f"Projects with repo_name {repo_name} deleted successfully."
            else:
                message = "Projects not found or already deleted."
            logging.info(message)
            return message
        except Exception as e:
            self.db.rollback()
            logging.error(f"An error occurred: {e}")
            return "Error occurred during deletion."


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

    def delete_project(db: Session, project_id: int):
        db.query(Project).filter(Project.id == project_id).delete()
        db.commit()


    def get_projects_by_repo_name(db: Session, repo_name: str, user_id: str, is_deleted: bool = False):
        try:
            projects = db.query(Project).filter(
                and_(
                    Project.repo_name == repo_name,
                    Project.user_id == user_id,
                    Project.is_deleted == is_deleted
                )
            ).all()

            return projects
        except Exception as e:
            db.rollback()
            # Log the error
            logging.error(f"Error fetching projects: {str(e)}")
            # You might want to raise a custom exception here instead of returning None
            return None



    
