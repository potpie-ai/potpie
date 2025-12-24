import logging
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import String, cast
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
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

    def _ensure_user_exists(self, user_id: str, user_email: str = None):
        """Ensure user exists in the database, create if not."""
        from app.modules.users.user_model import User
        from app.modules.users.user_schema import CreateUser
        from datetime import datetime
        
        # First check by uid
        user = self.db.query(User).filter(User.uid == user_id).first()
        if user:
            return user
        
        # If not found by uid, check by email (in case user exists with different uid)
        if user_email:
            user = self.db.query(User).filter(User.email == user_email).first()
            if user:
                logger.info(
                    f"User with email {user_email} exists with uid {user.uid}, "
                    f"but request has uid {user_id}. Using existing user {user.uid}."
                )
                # Return the existing user - the project will be associated with the existing user
                return user
        
        # User doesn't exist, create new one
        logger.info(f"User {user_id} not found in database, creating user record")
        user_data = CreateUser(
            uid=user_id,
            email=user_email or f"{user_id}@potpie.ai",
            display_name=user_id or "User",
            email_verified=False,
            created_at=datetime.utcnow(),
            last_login_at=datetime.utcnow(),
            provider_info={},
            provider_username="",  # Empty string for local/API-created users
        )
        from app.modules.users.user_service import UserService
        user_service = UserService(self.db)
        uid, message, error = user_service.create_user(user_data)
        if error:
            logger.error(f"Failed to create user {user_id}: {message}")
            raise Exception(f"Failed to create user: {message}")
        logger.info(f"Created user {user_id} in database")
        
        # Refresh to get the created user
        user = self.db.query(User).filter(User.uid == user_id).first()
        return user

    async def register_project(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        project_id: str,
        commit_id: str = None,
        repo_path: str = None,
        user_email: str = None,
    ):
        # Ensure user exists in database before checking project ownership
        # This may return a user with a different uid if user exists by email
        user = self._ensure_user_exists(user_id, user_email)
        # Use the actual user's uid from the database (may differ from token uid)
        actual_user_id = user.uid
        
        # Check if a project with this ID already exists
        existing_project = (
            self.db.query(Project).filter(Project.id == project_id).first()
        )

        if existing_project:
            # Check ownership using actual_user_id from database, not token user_id
            if existing_project.user_id != actual_user_id:
                message = (
                    f"Project {project_id} ownership mismatch: "
                    f"stored user {existing_project.user_id}, requesting user {actual_user_id} "
                    f"(token user_id: {user_id})"
                )
                logger.warning(message)
                raise HTTPException(status_code=403, detail=message)

            # Update the existing project with new information (e.g., normalized repo_name)
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
                self.db.commit()
                self.db.refresh(existing_project)
            except Exception:
                logger.exception(f"Error updating existing project {project_id}")
                self.db.rollback()
                raise
            message = f"Project id '{project_id}' for repo '{repo_name}' and branch '{branch_name}' updated successfully."
            logger.info(message)
            return project_id
        
        # Create new project if it doesn't exist
        project = Project(
            id=project_id,
            repo_name=repo_name,
            branch_name=branch_name,
            user_id=actual_user_id,  # Use actual user_id from database
            repo_path=repo_path,
            commit_id=commit_id,
            status=ProjectStatusEnum.SUBMITTED.value,
        )
        try:
            project = ProjectService.create_project(self.db, project)
        except Exception as e:
            logger.error(f"Error creating project {project_id}: {e}")
            self.db.rollback()
            raise
        message = f"Project id '{project.id}' for repo '{repo_name}' and branch '{branch_name}' registered successfully."
        logger.info(message)
        return project_id

    async def duplicate_project(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        project_id: str,
        properties,
        commit_id,
    ):
        project = Project(
            id=project_id,
            repo_name=repo_name,
            branch_name=branch_name,
            user_id=user_id,
            properties=properties,
            commit_id=commit_id,
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
        try:
            ProjectService.update_project(self.db, project_id, status=status.value)
            logger.info(
                f"Project with ID {project_id} has now been updated with status {status}."
            )
        except Exception:
            logger.exception(f"Error updating project status for {project_id}")
            self.db.rollback()
            raise

    async def get_project_from_db(
        self,
        repo_name: str,
        branch_name: str,
        user_id: str,
        repo_path: str | None = None,
        commit_id: str | None = None,
    ):
        """
        Get a project from the database for a specific user, prioritizing commit_id over branch_name.

        This method attempts to find an existing project by:
        1. First trying exact commit_id match (if commit_id provided)
        2. Falling back to branch_name match if commit_id match fails or not provided

        Args:
            repo_name: Repository name
            branch_name: Branch name (used as fallback if commit_id doesn't match)
            user_id: User ID
            repo_path: Path to the repository (optional)
            commit_id: Commit ID (optional, will try exact match first then fall back to branch)

        Returns:
            Project object if found by either commit_id or branch_name, None if no match
        """
        query = self.db.query(Project).filter(
            Project.repo_name == repo_name,
            Project.user_id == user_id,
            Project.repo_path == repo_path,
        )

        logger.info(
            f"Looking up project: repo_name={repo_name}, branch={branch_name}, "
            f"user={user_id}, repo_path={repo_path}, commit_id={commit_id}"
        )

        if commit_id:
            # If commit_id is provided, try to find exact match first
            project = query.filter(Project.commit_id == commit_id).first()
            if project:
                logger.info(f"Found project by commit_id: {project.id}")
                return project
            # ✅ FIX: Fall through to branch-based lookup instead of returning None
            logger.info(
                f"No project found with commit_id={commit_id}, falling back to branch lookup"
            )

        # Fall back to branch_name lookup
        project = query.filter(Project.branch_name == branch_name).first()
        if project:
            logger.info(f"Found project by branch_name: {project.id}")
        else:
            logger.info("No existing project found for this repository and branch")
        return project

    async def get_global_project_from_db(
        self,
        repo_name: str,
        branch_name: str,
        repo_path: str | None = None,
        commit_id: str | None = None,
    ):
        """
        Get a global project from the database, prioritizing commit_id over branch_name.

        Args:
            repo_name: Repository name
            branch_name: Branch name (used as fallback if commit_id is None)
            repo_path: Path to the repository (optional)
            commit_id: Commit ID (optional, prioritized over branch_name if provided)

        Returns:
            Project object if found, None otherwise
        """
        query = self.db.query(Project).filter(
            Project.repo_name == repo_name,
            Project.status == ProjectStatusEnum.READY.value,
            Project.repo_path == repo_path,
        )

        if commit_id:
            # If commit_id is provided, use it for deduplication
            project = (
                query.filter(Project.commit_id == commit_id)
                .order_by(Project.created_at.asc())
                .first()
            )
            if project:
                return project

        # Fall back to branch_name if commit_id is not provided or no match was found
        project = (
            query.filter(Project.branch_name == branch_name)
            .order_by(Project.created_at.asc())
            .first()
        )
        return project

    async def get_project_from_db_by_id(self, project_id: int):
        project = ProjectService.get_project_by_id(self.db, project_id)
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
        else:
            return None

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
        project = (
            self.db.query(Project)
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
        else:
            return None

    async def get_repo_and_branch_name(self, project_id: int):
        project = ProjectService.get_project_by_id(self.db, project_id)
        if project:
            return (
                project.repo_name,
                project.branch_name,
                project.directory,
                project.repo_path,
            )
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

    def update_project(db: Session, project_id: int, **kwargs):
        project = db.query(Project).filter(Project.id == project_id).first()

        if project is None:
            return None  # Project doesn't exist

        result = db.query(Project).filter(Project.id == project_id).update(kwargs)

        if result > 0:
            db.commit()
            return result

        return None

    async def delete_project(self, project_id: str):
        project = (
            self.db.query(Project)
            .filter(cast(Project.id, String) == str(project_id))
            .first()
        )
        if not project:
            raise HTTPException(status_code=404, detail="Project not found.")
        self.db.delete(project)
        self.db.commit()

    async def get_demo_project_id(self, repo_name: str):
        try:
            # Query for the project associated with the demo repo name
            project = (
                self.db.query(Project).filter(Project.repo_name == repo_name).first()
            )

            if project:
                logger.info(
                    f"Retrieved demo repo ID: {project.id} for repo name: {repo_name}"
                )
                return project.id  # Return the demo repo ID
            else:
                raise ProjectNotFoundError(
                    f"No demo repository found for repo name: {repo_name}"
                )

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
