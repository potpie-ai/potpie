import logging
import os
import shutil
import time
import traceback
from asyncio import create_task
from contextlib import contextmanager

from blar_graph.db_managers import Neo4jManager
from blar_graph.graph_construction.core.graph_builder import GraphConstructor
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config_provider import config_provider
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.parsing.graph_construction.parsing_helper import (
    ParseHelper,
    ParsingFailedError,
    ParsingServiceError,
)
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService
from app.modules.utils.email_helper import EmailHelper

from .parsing_schema import ParsingRequest


class ParsingService:
    def __init__(self, db: Session):
        self.db = db
        self.parse_helper = ParseHelper(db)
        self.project_service = ProjectService(db)
        self.inference_service = InferenceService()
        self.search_service = SearchService(db)

    @contextmanager
    def change_dir(self, path):
        old_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_dir)

    async def parse_directory(
        self,
        repo_details: ParsingRequest,
        user_id: str,
        user_email: str,
        project_id: int,
    ):
        # TODO: Cleanup Tech Debt
        project_manager = ProjectService(self.db)
        parse_helper = ParseHelper(self.db)
        extracted_dir = None

        try:
            repo, owner, auth = await parse_helper.clone_or_copy_repository(
                repo_details, self.db, user_id
            )
            extracted_dir, project_id = await parse_helper.setup_project_directory(
                repo, repo_details.branch_name, auth, repo, user_id, project_id
            )

            await self.analyze_directory(extracted_dir, project_id, user_id, self.db)

            message = "The project has been parsed successfully"
            await project_manager.update_project_status(
                project_id, ProjectStatusEnum.READY
            )
            create_task(EmailHelper().send_email(user_email))
            return {"message": message, "id": project_id}

        except ParsingServiceError as e:
            message = str(f"{project_id} Failed during parsing: " + e.message)
            await project_manager.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            raise HTTPException(status_code=500, detail=message)

        except Exception as e:
            await project_manager.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            tb_str = "".join(traceback.format_exception(None, e, e.__traceback__))
            raise HTTPException(
                status_code=500, detail=f"{str(e)}\nTraceback: {tb_str}"
            )

        finally:
            if extracted_dir:
                shutil.rmtree(extracted_dir, ignore_errors=True)

    async def analyze_directory(self, extracted_dir: str, project_id: int, user_id: str, db):
        logging.info(f"Analyzing directory: {extracted_dir}")
        repo_lang = self.parse_helper.detect_repo_language(extracted_dir)

        if repo_lang in ["javascript", "typescript"]:
            graph_manager = Neo4jManager(project_id, user_id)

            try:
                graph_constructor = GraphConstructor(graph_manager, user_id)
                n, r = graph_constructor.build_graph(extracted_dir)
                graph_manager.save_graph(n, r)

                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                # Create search index
                for node in n:
                    await self.search_service.create_search_index(
                        project_id, node["attributes"]
                    )
                await self.search_service.commit_indices()

                # Generate docstrings using InferenceService
                await self.inference_service.run_inference(project_id)

                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
            except Exception as e:
                logging.error(e)
                logging.error(traceback.format_exc())
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
            finally:
                graph_manager.close()
        elif repo_lang != "other":
            try:
                neo4j_config = config_provider.get_neo4j_config()
                service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    db,
                )

                await service.create_and_store_graph(extracted_dir, project_id, user_id)

                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                # Generate docstrings using InferenceService
                await self.inference_service.run_inference(project_id)

                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
            finally:
                service.close()
        else:
            await self.project_service.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            raise ParsingFailedError(
                "Repository doesn't consist of a language currently supported."
            )
