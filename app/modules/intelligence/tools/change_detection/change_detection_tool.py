import asyncio
import logging
from typing import Dict, List

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tree_sitter_languages import get_parser

from app.core.database import get_db
from app.modules.code_provider.code_provider_service import CodeProviderService
from app.modules.code_provider.github.github_service import GithubService
from app.modules.code_provider.local_repo.local_repo_service import LocalRepoService
from app.modules.intelligence.tools.kg_based_tools.get_code_from_node_id_tool import (
    GetCodeFromNodeIdTool,
)
from app.modules.parsing.graph_construction.parsing_repomap import RepoMap
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.projects.projects_service import ProjectService
from app.modules.search.search_service import SearchService


class ChangeDetectionInput(BaseModel):
    project_id: str = Field(
        ..., description="The ID of the project being evaluated, this is a UUID."
    )


class ChangeDetail(BaseModel):
    updated_code: str = Field(..., description="The updated code for the node")
    entrypoint_code: str = Field(..., description="The code for the entry point")
    citations: List[str] = Field(
        ..., description="List of file names referenced in the response"
    )


class ChangeDetectionResponse(BaseModel):
    patches: Dict[str, str] = Field(..., description="Dictionary of file patches")
    changes: List[ChangeDetail] = Field(
        ..., description="List of changes with updated and entry point code"
    )


class ChangeDetectionTool:
    name = "Get code changes"
    description = """Analyzes differences between branches in a Git repository and retrieves updated function details.
        :param project_id: string, the ID of the project being evaluated (UUID).

            example:
            {
                "project_id": "550e8400-e29b-41d4-a716-446655440000"
            }

        Returns dictionary containing:
        - patches: Dict[str, str] - file patches
        - changes: List[ChangeDetail] - list of changes with updated and entry point code
        """

    def __init__(self, sql_db, user_id):
        self.sql_db = sql_db
        self.user_id = user_id
        self.search_service = SearchService(self.sql_db)

    def _parse_diff_detail(self, patch_details):
        changed_files = {}
        current_file = None
        for filename, patch in patch_details.items():
            lines = patch.split("\n")
            current_file = filename
            changed_files[current_file] = set()
            for line in lines:
                if line.startswith("@@"):
                    parts = line.split()
                    add_start_line, add_num_lines = (
                        map(int, parts[2][1:].split(","))
                        if "," in parts[2]
                        else (int(parts[2][1:]), 1)
                    )
                    for i in range(add_start_line, add_start_line + add_num_lines):
                        changed_files[current_file].add(i)
        return changed_files

    async def _find_changed_functions(self, changed_files, project_id):
        result = []
        for relative_file_path, lines in changed_files.items():
            try:
                project = await ProjectService(self.sql_db).get_project_from_db_by_id(
                    project_id
                )
                code_service = CodeProviderService(self.sql_db)
                file_content = code_service.get_file_content(
                    project["project_name"],
                    relative_file_path,
                    0,
                    0,
                    project["branch_name"],
                    project_id,
                )
                tags = RepoMap.get_tags_from_code(relative_file_path, file_content)

                language = RepoMap.get_language_for_file(relative_file_path)
                if language:
                    parser = get_parser(language.name)
                    tree = parser.parse(bytes(file_content, "utf8"))
                    root_node = tree.root_node

                nodes = {}
                for tag in tags:
                    if tag.kind == "def":
                        if tag.type == "class":
                            node_type = "CLASS"
                        elif tag.type in ["method", "function"]:
                            node_type = "FUNCTION"

                        else:
                            node_type = "other"

                        node_name = f"{relative_file_path}:{tag.name}"

                        if language:
                            node = RepoMap.find_node_by_range(
                                root_node, tag.line, node_type
                            )
                        if node:
                            nodes[node_name] = node

                for node_name, node in nodes.items():
                    start_line = node.start_point[0]
                    end_line = node.end_point[0]
                    if any(start_line < line < end_line for line in lines):
                        result.append(node_name)
            except Exception as e:
                logging.error(f"Exception {e}")
        return result

    async def get_updated_function_list(self, patch_details, project_id):
        changed_files = self._parse_diff_detail(patch_details)
        return await self._find_changed_functions(changed_files, project_id)

    @staticmethod
    def _find_inbound_neighbors(tx, node_id, project_id, with_bodies):
        query = f"""
        MATCH (start:Function {{id: $endpoint_id, project_id: $project_id}})
        CALL {{
            WITH start
            MATCH (neighbor:Function {{project_id: $project_id}})-[:CALLS*]->(start)
            RETURN neighbor{', neighbor.body AS body' if with_bodies else ''}
        }}
        RETURN start, collect({{neighbor: neighbor{', body: neighbor.body' if with_bodies else ''}}}) AS neighbors
        """
        endpoint_id = node_id
        result = tx.run(query, endpoint_id, project_id)
        record = result.single()
        if not record:
            return []

        start_node = dict(record["start"])
        neighbors = record["neighbors"]
        combined = [start_node] + neighbors if neighbors else [start_node]
        return combined

    def traverse(self, identifier, project_id, neighbors_fn):
        neighbors_query = neighbors_fn(with_bodies=False)
        with self.driver.session() as session:
            return session.read_transaction(
                self._traverse, identifier, project_id, neighbors_query
            )

    def find_entry_points(self, identifiers, project_id):
        all_inbound_nodes = set()

        for identifier in identifiers:
            traversal_result = self.traverse(
                identifier=identifier,
                project_id=project_id,
                neighbors_fn=ChangeDetectionTool._find_inbound_neighbors,
            )
            for item in traversal_result:
                if isinstance(item, dict):
                    all_inbound_nodes.update([frozenset(item.items())])

        entry_points = set()
        for node in all_inbound_nodes:
            node_dict = dict(node)
            traversal_result = self.traverse(
                identifier=node_dict["id"],
                project_id=project_id,
                neighbors_fn=ChangeDetectionTool._find_inbound_neighbors,
            )
            if len(traversal_result) == 1:
                entry_points.add(node)

        return entry_points

    async def get_code_changes(self, project_id):
        global patches_dict, repo
        patches_dict = {}
        project_details = await ProjectService(self.sql_db).get_project_from_db_by_id(
            project_id
        )

        if project_details is None:
            raise HTTPException(status_code=400, detail="Project Details not found.")

        if project_details["user_id"] != self.user_id:
            raise ValueError(
                f"Project id {project_id} not found for user {self.user_id}"
            )

        repo_name = project_details["project_name"]
        branch_name = project_details["branch_name"]
        repo_path = project_details["repo_path"]
        # Use CodeProviderService to get the appropriate service instance
        code_service = CodeProviderService(self.sql_db)
        try:
            if isinstance(code_service.service_instance, GithubService):
                github, _, _ = code_service.service_instance.get_github_repo_details(
                    repo_name
                )
                repo = github.get_repo(repo_name)
                default_branch = repo.default_branch
                git_diff = repo.compare(default_branch, branch_name)
                patches_dict = {
                    file.filename: file.patch for file in git_diff.files if file.patch
                }
            elif isinstance(code_service.service_instance, LocalRepoService):
                patches_dict = code_service.service_instance.get_local_repo_diff(
                    repo_path, branch_name
                )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error while fetching changes: {str(e)}"
            )
        finally:
            if project_details is not None:
                identifiers = []
                node_ids = []
                try:
                    identifiers = await self.get_updated_function_list(
                        patches_dict, project_id
                    )
                    for identifier in identifiers:
                        node_id_query = " ".join(identifier.split(":"))
                        relevance_search = await self.search_service.search_codebase(
                            project_id, node_id_query
                        )
                        if relevance_search:
                            node_id = relevance_search[0]["node_id"]
                            if node_id:
                                node_ids.append(node_id)
                        else:
                            node_ids.append(
                                GetCodeFromNodeIdTool(self.sql_db, self.user_id).run(
                                    project_id, identifier
                                )["node_id"]
                            )

                    # Fetch code for node ids and store in a dict
                    node_code_dict = {}
                    for node_id in node_ids:
                        node_code = GetCodeFromNodeIdTool(
                            self.sql_db, self.user_id
                        ).run(project_id, node_id)
                        node_code_dict[node_id] = {
                            "code_content": node_code["code_content"],
                            "file_path": node_code["file_path"],
                        }

                    entry_points = InferenceService(
                        self.sql_db, "dummy"
                    ).get_entry_points_for_nodes(node_ids, project_id)

                    changes_list = []
                    for node, entry_point in entry_points.items():
                        entry_point_code = GetCodeFromNodeIdTool(
                            self.sql_db, self.user_id
                        ).run(project_id, entry_point[0])
                        changes_list.append(
                            ChangeDetail(
                                updated_code=node_code_dict[node]["code_content"],
                                entrypoint_code=entry_point_code["code_content"],
                                citations=[
                                    node_code_dict[node]["file_path"],
                                    entry_point_code["file_path"],
                                ],
                            )
                        )

                    return ChangeDetectionResponse(
                        patches=patches_dict, changes=changes_list
                    )
                except Exception as e:
                    logging.error(f"project_id: {project_id}, error: {str(e)}")

                if len(identifiers) == 0:
                    return []

    async def arun(self, project_id: str) -> str:
        return await self.get_code_changes(project_id)

    def run(self, project_id: str) -> str:
        return asyncio.run(self.get_code_changes(project_id))


def get_change_detection_tool(user_id: str) -> StructuredTool:
    """
    Get a list of LangChain Tool objects for use in agents.
    """
    change_detection_tool = ChangeDetectionTool(next(get_db()), user_id)
    return StructuredTool.from_function(
        coroutine=change_detection_tool.arun,
        func=change_detection_tool.run,
        name="Get code changes",
        description="""
            Get the changes in the codebase.
            This tool analyzes the differences between branches in a Git repository and retrieves updated function details, including their entry points and citations.
            Inputs for the get_code_changes method:
            - project_id (str): The ID of the project being evaluated, this is a UUID.
            The output includes a dictionary of file patches and a list of changes with updated code and entry point code.
            """,
        args_schema=ChangeDetectionInput,
    )
