import asyncio
import logging
import os
from typing import Any, Dict

from app.celery.celery_app import celery_app
from app.celery.tasks.base_task import BaseTask
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.projects.projects_service import ProjectService
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.utils.email_helper import EmailHelper
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.core.database import get_db
from app.core.config_provider import config_provider

logger = logging.getLogger(__name__)



@celery_app.task(
    bind=True,
    base=BaseTask,
    name="app.celery.tasks.parsing_tasks.process_parsing",
)
def process_parsing(
    self,
    repo_details: Dict[str, Any],
    user_id: str,
    user_email: str,
    project_id: str,
    cleanup_graph: bool = True,
) -> None:
    logger.info(f"Task received: Starting parsing process for project {project_id}")
        
    # Clean the input dictionary by removing None and empty values
    cleaned_repo_details = {k: v for k, v in repo_details.items() if v is not None and v != ""}
    logger.info(f"Cleaned repo_details: {cleaned_repo_details}")
    try:
        # Create ParsingRequest object from cleaned data
        parsing_request = ParsingRequest(**cleaned_repo_details)
        logger.info(f"Created ParsingRequest: repo_name={parsing_request.repo_name}, repo_path={parsing_request.repo_path}")
        
        # Determine if this is a local or remote repository
        is_local = False
        is_remote = False
        
        if parsing_request.repo_path and os.path.exists(parsing_request.repo_path):
            is_local = True
            logger.info(f"Detected local repository: {parsing_request.repo_path}")
        elif parsing_request.repo_name:
            is_remote = True
            logger.info(f"Detected remote repository: {parsing_request.repo_name}")
        else:
            logger.error("Neither valid local path nor remote repo name found")
            raise ValueError("Either a valid local repository path or remote repository name must be provided")
        
        # Run the async parsing function
        asyncio.run(run_parsing_async(
            parsing_request, user_id, user_email, project_id, cleanup_graph, is_local, is_remote
        ))
        
    except Exception as e:
        logger.error(f"Error in process_parsing task: {e}")
        # Try to update project status to failed
        try:
            db = next(get_db())
            project_service = ProjectService(db)
            # Try to update status to ERROR
            try:
                asyncio.run(project_service.update_project_status(int(project_id), ProjectStatusEnum.ERROR))
            except ValueError:
                # If project_id can't be converted to int, try with string
                asyncio.run(project_service.update_project_status(project_id, ProjectStatusEnum.ERROR))
            db.close()
        except Exception as status_error:
            logger.error(f"Failed to update project status to ERROR: {status_error}")
        raise


async def run_parsing_async(parsing_request: ParsingRequest, user_id: str, user_email: str, project_id: str, 
                           cleanup_graph: bool, is_local: bool, is_remote: bool):
    """
    Async function to handle the parsing process
    """
    # Get database session
    db = next(get_db())
    
    try:
        # Initialize services
        project_service = ProjectService(db)
        parsing_service = ParsingService(db, user_id)
        code_provider_service = CodeProviderService(db)
        
        # The parsing service expects int, but project service expects str for some methods
        # Let's check if we can convert the UUID to int, if not, we need to use string methods
        try:
            project_id_int = int(project_id)
            logger.info(f"Successfully converted project_id to int: {project_id_int}")
        except ValueError:
            logger.info(f"Project ID is UUID string, cannot convert to int: {project_id}")
            # For now, let's use the string version and see what happens
            project_id_int = project_id
        
        # Update project status to submitted (indicating processing has started) - this might need string ID
        try:
            await project_service.update_project_status(project_id_int, ProjectStatusEnum.SUBMITTED)
        except Exception as e:
            logger.error(f"Failed to update project status with int ID, trying string ID: {e}")
            # Try with string ID if int fails
            await project_service.update_project_status(project_id, ProjectStatusEnum.SUBMITTED)
        
        if cleanup_graph:
            logger.info(f"Cleaning up existing graph for project {project_id}")
            neo4j_config = config_provider.get_neo4j_config()
            code_graph_service = CodeGraphService(
                neo4j_config["uri"],
                neo4j_config["username"],
                neo4j_config["password"],
                db
            )
            code_graph_service.cleanup_graph(project_id)
        
        # Use the parsing service's parse_directory method which handles both local and remote repos
        logger.info(f"Starting parsing with service for project {project_id}")
        await parsing_service.parse_directory(
            parsing_request,
            user_id,
            user_email,
            project_id_int,
            cleanup_graph
        )
        
        # Update project status to ready
        try:
            await project_service.update_project_status(project_id_int, ProjectStatusEnum.READY)
        except Exception as e:
            logger.error(f"Failed to update project status with int ID, trying string ID: {e}")
            await project_service.update_project_status(project_id, ProjectStatusEnum.READY)
        
        # Send completion email
        if user_email:
            await EmailHelper().send_email(user_email, parsing_request.repo_name or "Repository", parsing_request.branch_name)
        
        logger.info(f"Parsing completed successfully for project {project_id}")
        
    except Exception as e:
        logger.error(f"Error during parsing for project {project_id}: {e}")
        try:
            await project_service.update_project_status(project_id_int, ProjectStatusEnum.ERROR)
        except Exception as status_error:
            logger.error(f"Failed to update project status with int ID, trying string ID: {status_error}")
            try:
                await project_service.update_project_status(project_id, ProjectStatusEnum.ERROR)
            except Exception as final_error:
                logger.error(f"Failed to update project status with both int and string ID: {final_error}")
        raise
    finally:
        db.close()


logger.info("Parsing tasks module loaded")
