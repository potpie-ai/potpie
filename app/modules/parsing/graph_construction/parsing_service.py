import logging
import traceback

from blar_graph.db_managers import Neo4jManager
from blar_graph.graph_construction.core.graph_builder import GraphConstructor

from app.core.config import config_provider
from app.modules.parsing.graph_construction.code_graph_service import CodeGraphService
from app.modules.parsing.graph_construction.parsing_helper import (
    ParseHelper,
    ParsingFailedError,
)
from app.modules.projects.projects_schema import ProjectStatusEnum
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService


class ParsingService:
    # @celery_worker_instance.celery_instance.task(name='app.modules.parsing.graph_construction.parsing_service.analyze_directory')
    @staticmethod
    async def analyze_directory(extracted_dir: str, project_id: int, user_id: str, db):
        logging.info(f"Analyzing directory: {extracted_dir}")

        try:
            await ParsingService._analyze_directory(
                extracted_dir, project_id, user_id, db
            )
        finally:
            db.close()

    async def _analyze_directory(extracted_dir: str, project_id: int, user_id: str, db):
        logging.info(f"_Analyzing directory: {extracted_dir}")
        repo_lang = ParseHelper(db).detect_repo_language(extracted_dir)

        if repo_lang in ["python", "javascript", "typescript"]:
            graph_manager = Neo4jManager(project_id, user_id)

            try:
                graph_constructor = GraphConstructor(graph_manager, user_id)
                n, r = graph_constructor.build_graph(extracted_dir)
                graph_manager.save_graph(n, r)

                # Create search index
                search_service = SearchService(db)
                for node in n:
                    await search_service.create_search_index(
                        project_id, node["attributes"]
                    )

                await ProjectService(db).update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
            except Exception as e:
                logging.error(e)
                logging.error(traceback.format_exc())
                await ProjectService(db).update_project_status(
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

                service.create_and_store_graph(extracted_dir, project_id, user_id)
                await ProjectService(db).update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
            finally:
                service.close()
        else:
            await ProjectService(db).update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
            return ParsingFailedError(
                "Repository doesn't consist of a language currently supported."
            )
